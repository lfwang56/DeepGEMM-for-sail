"""PPU1.5 W4FA8 functional tests. Run: python tests/test_w4a8.py"""
import random
from contextlib import contextmanager

import torch
import deep_gemm
from utils import is_ppu1v5_device
from deep_gemm.testing.numeric import calc_diff
from quantization import decode_weight, quant_activation
import utils


@contextmanager
def capture_after_warmup(graph, call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        yield stream


def test_grouped_gemm():
    cases = [
        'GroupedNoPad,data_type:w4fa8_mma,groups:4,m:35,n:64,k:128,quant_type:group,group_size:32',
        'GroupedNoPad,data_type:w4fa8_mma,groups:128,m:7,n:192,k:256,quant_type:group,group_size:32',
        'GroupedMasked,data_type:w4fa8_mma,groups:4,m:67,n:192,k:256,distribution:[0,1,33,67],quant_type:group,group_size:32',
        'GroupedMasked,data_type:w4fa8_mma,groups:4,m:9,n:192,k:512,em:8,distribution:[0,1,8,9],quant_type:group,group_size:32',
        'GroupedMasked,data_type:w4fa8_mma,groups:128,m:35,n:128,k:512,em:32,quant_type:group,group_size:32',
        'GroupedFused,data_type:w4fa8_mma,groups:8,num_token:17,topk:3,n:192,k:256,quant_type:group,group_size:32',
        'GroupedFused,data_type:w4fa8_mma,groups:128,num_token:4,topk:2,n:64,k:2048,quant_type:group,group_size:32',
    ]
    runners = {
        'GroupedNoPad': utils.test_m_grouped_gemm_nopad,
        'GroupedMasked': utils.test_m_grouped_gemm_masked,
        'GroupedFused': utils.test_m_grouped_gemm_fused,
    }
    utils.set_acc_check(True)
    utils.set_benchmark(False)
    for case in cases:
        print(case)
        args = utils.parse_deepgemm_string_re(case)
        runners[args['gemm_type']](args)


def test_inv_scale():
    def compare(call, lhs, shape, name, valid_mask=None):
        for inverse in (1.0, .03125, .3, 2.0):
            reference = torch.full(shape, -123., device='cuda', dtype=torch.bfloat16)
            output = torch.full_like(reference, -123.)
            call((lhs[0], lhs[1] * inverse), reference)
            call(lhs, output, inv_scale=inverse)
            assert torch.equal(output, reference), (name, inverse)
            if valid_mask is not None:
                assert (output[~valid_mask] == -123.).all(), (name, inverse, "padding overwritten")
        for inverse in (0., -1., float('inf'), float('nan')):
            try:
                call(lhs, output, inv_scale=inverse)
            except RuntimeError as error:
                assert 'inv_scale must be finite and positive' in str(error)
            else:
                raise AssertionError(f'accepted invalid {inverse=}')
        print(f'PASS inv_scale {name}: folded-scale equivalence and invalid values')

    counts = [0, 1, 17, 65]
    rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
    indices = torch.repeat_interleave(torch.arange(4, device='cuda', dtype=torch.int32), rows.long())
    sms = deep_gemm.get_num_sms()
    # Cover uncached scales, resident A with cached scales, and non-expanded tiles.
    for k, config in [(128, (sms, 16, 256, 128, 16, 64, 128, 2, 1)),
                      (384, (sms, 64, 256, 128, 64, 64, 128, 2, 3)),
                      (256, (sms, 32, 128, 128, 32, 64, 128, 2, 1))]:
        n = 768
        packed = torch.randint(256, (4, n, k//2), device='cuda', dtype=torch.uint8)
        raw = torch.randint(109, 125, (4, n, k//32), device='cuda', dtype=torch.uint8)
        raw[0, 0, 0] = 109
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        assert rhs[2] == .03125
        lhs = quant_activation(torch.randn(sum(counts), k, device='cuda', dtype=torch.bfloat16))
        def nopad(a, out, **kwargs):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(a, rhs[:2], out, indices, rows, config, **kwargs)
        compare(nopad, lhs, (sum(counts), n), f'nopad {k=} {config=}')
        out = torch.empty(sum(counts), n, device='cuda', dtype=torch.bfloat16)
        nopad(lhs, out, inv_scale=rhs[2])
        start = 0
        for expert, count in enumerate(counts):
            if count:
                a = lhs[0][start:start+count].float() * lhs[1][start:start+count]
                ref = (a @ decode_weight(packed[expert], raw[expert]).T).bfloat16()
                assert calc_diff(out[start:start+count], ref) < .0015
            start += count

    lhs, rhs, routes, out, _ = utils.construct_non_permute_grouped(
        8, 17, 256, 192, 3, 'w4fa8_mma', 'group', 32)
    config, counts, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
    def fused(a, output, **kwargs):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            a, rhs, output, counts, experts, tokens, aligned, config, **kwargs)
    compare(fused, lhs, tuple(out.shape), 'fused gather and repeated token use')

    k, n, capacity = 512, 192, 67
    packed = torch.randint(256, (4, n, k//2), device='cuda', dtype=torch.uint8)
    raw = torch.full((4, n, k//32), 127, device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    lhs = quant_activation(torch.randn(4, capacity, k, device='cuda', dtype=torch.bfloat16))
    rows = torch.tensor([0, 1, 33, 67], device='cuda', dtype=torch.int32)
    def masked(a, output, **kwargs):
        config = (sms, 32, 128, 128, 32, 32, 128, 2, 1)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked(a, rhs[:2], output, rows, 25, config, **kwargs)
    valid = torch.arange(capacity, device='cuda')[None, :] < rows[:, None]
    compare(masked, lhs, (4, capacity, n), 'masked valid rows and untouched padding', valid)


def test_nopad_expert_limit():
    # The count-preprocessing kernel has storage for at most 32 warps.
    for groups in (1024, 1025):
        packed = torch.zeros(groups, 64, 64, device='cuda', dtype=torch.uint8)
        raw = torch.full((groups, 64, 4), 127, device='cuda', dtype=torch.uint8)
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        lhs = (torch.zeros(1, 128, device='cuda', dtype=torch.float8_e4m3fn),
               torch.ones(1, 1, device='cuda'))
        rows = torch.zeros(groups, device='cuda', dtype=torch.int32)
        rows[-1] = 1
        indices = torch.full((1,), groups - 1, device='cuda', dtype=torch.int32)
        out = torch.full((1, 64), float('nan'), device='cuda', dtype=torch.bfloat16)
        if groups == 1024:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows)
            assert torch.equal(out, torch.zeros_like(out))
        else:
            try:
                deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows)
            except RuntimeError as error:
                assert 'at most 1024 experts' in str(error)
            else:
                raise AssertionError('accepted 1025 nopad experts')
    print('PASS nopad 1024 experts and rejection of 1025 experts')


def test_nopad_n_expand():
    # Short-K pipeline boundaries, three-way expansion, N tails and long-K fallback.
    for groups, n, k, expand in [(4, 512, 128, 2), (4, 768, 128, 3),
                                  (128, 1024, 128, 4), (4, 768, 512, 3),
                                  (4, 1024, 512, 4), (4, 320, 128, 1),
                                  (4, 1024, 768, 1)]:
        counts = [0, 1, 17, 65] + [0] * (groups - 4)
        rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
        indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
        m = sum(counts)
        lhs = quant_activation(torch.randn(m, k, device='cuda', dtype=torch.bfloat16))
        packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
        raw = torch.randint(125, 130, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        gemm_lhs = (lhs[0], lhs[1] * rhs[2])
        bm = 16
        while bm < (m + groups - 1) // groups and bm < 64:
            bm *= 2
        config = (deep_gemm.get_num_sms(), bm, 256, 128, bm, 64, 128, 2, 1)
        baseline = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(baseline)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], baseline, indices, rows, config)
        offset = 0
        for expert, count in enumerate(counts):
            if count:
                a = lhs[0][offset:offset+count].float() * lhs[1][offset:offset+count]
                ref = (a @ decode_weight(packed[expert], raw[expert]).T).bfloat16()
                assert torch.isfinite(baseline[offset:offset+count]).all()
                assert calc_diff(baseline[offset:offset+count], ref) < .0015
            offset += count
        for selected in (None, config[:-1] + (expand,)):
            out.fill_(float('nan'))
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, selected)
            assert torch.equal(out, baseline), (groups, n, k, expand)
        if bm == 32 and k <= 512 and expand > 1:
            out.fill_(float('nan'))
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                gemm_lhs, rhs[:2], out, indices, rows, config[:-1] + (expand,), inv_scale=.25)
            assert torch.equal(out, baseline * .25), (groups, n, k, expand)
        # Explicit expansion remains valid above the automatic short-K threshold.
        if k == 768:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, config[:-1] + (4,))
            assert torch.equal(out, baseline)
        if n == 320:
            try:
                deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                    gemm_lhs, rhs[:2], out, indices, rows, config[:-1] + (2,))
            except RuntimeError:
                pass
            else:
                raise AssertionError('accepted N_EXPAND with unaligned N')
        print(f'PASS N_EXPAND {groups=} {n=} {k=} {expand=}')


def test_nopad_resident_a_and_scales():
    # N/K-varying weights and all legal scales expose reuse of the wrong shared
    # stage or output-row scale across successive N tiles and scheduler items.
    groups, n = 4, 768
    counts = [0, 1, 65, 197]
    m = sum(counts)
    rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
    indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
    for k in (128, 256, 384, 512):
        codes = ((torch.arange(n, device='cuda')[:, None]*3 + torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).expand(groups, n, k//2).contiguous()
        raw = (114 + (torch.arange(groups, device='cuda')[:, None, None] +
                      torch.arange(n, device='cuda')[None, :, None] +
                      torch.arange(k//32, device='cuda')[None, None, :]) % 27).byte().contiguous()
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        col = (torch.arange(m, device='cuda')*7) % k
        a = torch.zeros(m, k, device='cuda')
        a[torch.arange(m, device='cuda'), col] = 1
        scales = torch.linspace(.5, 1.5, m, device='cuda').reshape(-1, 1)
        lhs = (a.to(torch.float8_e4m3fn), scales)
        gemm_lhs = (lhs[0], lhs[1] * rhs[2])
        configs = [None]
        if k == 256:
            configs.append((deep_gemm.get_num_sms(), 64, 256, 128, 64, 64, 128, 2, 3))
        for config in configs:
            out = torch.full((m, n), float('nan'), device='cuda', dtype=torch.bfloat16)
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, config)
            start = 0
            for expert, count in enumerate(counts):
                if count:
                    expected = (decode_weight(packed[expert], raw[expert])[:, col[start:start+count]].T *
                                scales[start:start+count]).bfloat16()
                    assert torch.equal(out[start:start+count], expected), (k, expert, config)
                start += count
        print(f'PASS resident A/scale basis {k=}')


def test_fused_nopad_equivalence():
    case = 'GroupedFused,data_type:w4fa8_mma,groups:8,num_token:17,topk:3,n:192,k:256,quant_type:group,group_size:32'
    print(f'Fused/nopad equivalence: {case}')
    args = utils.parse_deepgemm_string_re(case)
    lhs, rhs, routes, out, _ = utils.construct_non_permute_grouped(
        args['groups'], args['num_token'], args['k'], args['n'], args['topk'],
        args['data_type'], args['quant_type'], args['group_size'])
    config, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(lhs, rhs, out, rows, experts, tokens, aligned, config)
    flat_experts = routes.flatten()
    order = flat_experts.argsort(stable=True)
    source = torch.arange(routes.size(0), device=routes.device).repeat_interleave(routes.size(1))[order]
    nopad_lhs = (lhs[0].float()[source].to(lhs[0].dtype), lhs[1][source].contiguous())
    nopad_out = torch.empty_like(out)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(nopad_lhs, rhs, nopad_out, flat_experts[order], rows)
    diff = calc_diff(out, nopad_out)
    assert torch.isfinite(out).all() and diff < .0015, f'fused/nopad {diff=}'
    print('PASS fused/nopad equivalence')


def test_boundaries():
    packed = torch.zeros(1, 64, 64, device='cuda', dtype=torch.uint8)
    scale = torch.full((1, 64, 4), 127, device='cuda', dtype=torch.uint8)
    for low, high in ((0, 127), (127, 255), (113, 141), (1, 254)):
        scale.fill_(high)
        scale[0, 0, 0] = low
        try:
            deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, None, scale)
        except RuntimeError as error:
            assert f'[{low},{high}]' in str(error) and '[114,140]' in str(error)
        else:
            raise AssertionError(f'accepted invalid scale range [{low},{high}]')
    for original, adjusted, inverse in ((113, 114, .5), (114, 114, 1.),
                                        (140, 140, 1.), (141, 140, 2.),
                                        (254, 140, float(2 ** 114))):
        scale.fill_(original)
        result = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scale)
        assert result[1].shape == (1, 1, 256)
        assert (result[1] == adjusted).all() and result[2] == inverse
    scale.fill_(127)
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scale)
    a = (torch.empty(0, 128, device='cuda', dtype=torch.float8_e4m3fn), torch.empty(0, 1, device='cuda'))
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(a, rhs[:2], torch.empty(0, 64, device='cuda', dtype=torch.bfloat16),
                                      torch.empty(0, device='cuda', dtype=torch.int32))
    # Catch dtype/scale shape mistakes at the public boundary, before any launch.
    a = (torch.zeros(1, 128, device='cuda', dtype=torch.float8_e4m3fn), torch.ones(1, 2, device='cuda'))
    try:
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(a, rhs[:2], torch.empty(1, 64, device='cuda', dtype=torch.bfloat16),
                                          torch.zeros(1, device='cuda', dtype=torch.int32))
    except RuntimeError:
        pass
    else:
        raise AssertionError('accepted invalid activation scale shape')
    # Scale-only preparation derives alignment from [E,N,K/32] before reducing exponents.
    for bad in (None, scale.float(), scale[0], scale[:, :63], scale[..., :3],
                scale[:0], scale[:, :0], scale[..., :0]):
        try:
            deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('accepted invalid scale-only W4A8 input')
    for mode, fuse in ((deep_gemm.PreprocessMode.MXFP4, True),
                       (deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, True),
                       (deep_gemm.PreprocessMode.W4A16_INT4, False),
                       (deep_gemm.PreprocessMode.W4A16_MXFP4, False),
                       (deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False)):
        try:
            deep_gemm.preprocess_weights_and_scales(mode, fuse, weight_scale=scale)
        except RuntimeError as error:
            assert 'requires weights' in str(error)
        else:
            raise AssertionError('accepted missing required weights')
    print('PASS preprocessing boundaries, empty input, invalid scale shape')


def test_tensor_scale_shift():
    # Folding the inverse must not change the activation codes.
    x = torch.randn(7, 128, device='cuda', dtype=torch.bfloat16)
    aq, a_scale = quant_activation(x)
    for inverse in (.03125, 32.):
        folded_aq, folded_scale = quant_activation(x, inverse)
        assert torch.equal(aq.view(torch.uint8), folded_aq.view(torch.uint8))
        torch.testing.assert_close(folded_scale, a_scale * inverse, rtol=0, atol=0)
    # The offset is shared across experts; compare the logical N/K scale order.
    e, n, k = 2, 128, 256
    packed = torch.randint(256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
    for low, high, shift in ((109, 124, 5), (111, 127, 3),
                             (119, 145, -5), (114, 140, 0)):
        raw = torch.full((e, n, k // 32), high, device='cuda', dtype=torch.uint8)
        raw[0, 0, 0] = low
        weight, adjusted, inverse = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        assert torch.equal(weight, packed)
        no_weight, scale_only, inverse_only = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, None, raw)
        assert no_weight is None and inverse_only == inverse
        assert torch.equal(scale_only, adjusted) and scale_only.stride() == adjusted.stride()
        no_weight, scale_only, inverse_only = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=raw)
        assert no_weight is None and inverse_only == inverse and torch.equal(scale_only, adjusted)
        assert inverse == float(2. ** -shift)
        assert adjusted.min().item() == low + shift
        assert adjusted.max().item() == high + shift
        expected = torch.empty(e, n // 64, k * 2, dtype=torch.uint8)
        host = raw.cpu()
        for expert in range(e):
            for tile in range(n // 64):
                for kg in range(k // 64):
                    for i in range(8):
                        for j in range(8):
                            for half in range(2):
                                index = kg * 128 + i * 16 + j * 2 + half
                                expected[expert, tile, index] = host[
                                    expert, tile * 64 + i + 8 * j, kg * 2 + half] + shift
        assert torch.equal(adjusted.cpu(), expected)
        decoded = decode_weight(packed, raw)
        adjusted_natural = (raw.short() + shift).byte()
        torch.testing.assert_close(decode_weight(packed, adjusted_natural) * inverse, decoded, rtol=0, atol=0)
    print('PASS batch-wide exponent shift, inverse scale, and layout')


def test_preprocessing_layout():
    # Explicit coordinate mapping, independent of the implementation's reshape/index_select.
    e, n, k = 2, 128, 256
    packed = torch.randint(256, (e, n, k//2), device='cuda', dtype=torch.uint8)
    raw = (torch.arange(e*n*k//32, device='cuda').reshape(e, n, k//32) % 27 + 114).byte()
    weight, actual, inverse = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert inverse == 1.0
    expected = torch.empty(e, n//64, k*2, dtype=torch.uint8)
    host = raw.cpu()
    for expert in range(e):
        for tile in range(n//64):
            for kg in range(k//64):
                for i in range(8):
                    for j in range(8):
                        for half in range(2):
                            expected[expert, tile, kg*128+i*16+j*2+half] = host[expert, tile*64+i+8*j, kg*2+half]
    assert torch.equal(weight, packed)
    assert torch.equal(actual.cpu(), expected)
    # W4FA16 MMA consumes the same byte layout, without W4FA8's scale window.
    weight16, scale16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw)
    assert torch.equal(weight16, packed)
    assert torch.equal(scale16, actual)
    raw16 = raw.clone()
    raw16[0, 0, 0] = 113
    raw16[1, 1, 1] = 141
    _, scale16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw16)
    assert scale16[0, 0, 0].item() == 113
    assert scale16[1, 0, 17].item() == 141
    packed64 = packed[..., :32].contiguous()
    raw64 = raw[..., :2].contiguous()
    weight64, scale64 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed64, raw64)
    assert torch.equal(weight64, packed64)
    assert torch.equal(scale64.cpu(), expected[..., :128])
    # Existing MXFP4 activation-fusion mode retains its interleaving and scale layout.
    weight, scale = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.MXFP4, True, packed, raw)
    interleaved_weight = torch.stack((packed[:, :n//2], packed[:, n//2:]), 2).flatten(1, 2)
    interleaved_scale = torch.stack((raw[:, :n//2], raw[:, n//2:]), 2).flatten(1, 2)
    assert torch.equal(weight, interleaved_weight)
    assert torch.equal(scale, deep_gemm.preprocess_mxfp4_scales(interleaved_scale))
    legacy_weight, legacy_scale = deep_gemm.preprocess_mxfp4_weight_for_act_and_quant_fusing(packed, raw)
    assert torch.equal(legacy_weight, weight) and torch.equal(legacy_scale, scale)
    print('PASS unified preprocessing layouts and existing MXFP4 mode')


def test_basis():
    # Each row selects one K coordinate. All weight codes and alternating group scales
    # expose layout swaps even when aggregate random-GEMM error is small.
    n, k = 64, 128
    codes = ((torch.arange(n, device='cuda')[:, None]*3 + torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).contiguous()
    aq = torch.eye(k, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.linspace(.5, 1.5, k, device='cuda').reshape(k, 1)
    out = torch.empty(k, n, device='cuda', dtype=torch.bfloat16)
    indices = torch.zeros(k, device='cuda', dtype=torch.int32)
    base = torch.tensor([114, 127, 130, 140], device='cuda', dtype=torch.int16)
    for offset in (-5, 0, 5):
        raw = (base + offset).byte().expand(1, n, 4).contiguous()
        rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad((aq, scales * rhs[2]), rhs[:2], out, indices)
        expected = (decode_weight(packed[0], raw[0]).T * scales).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    raw = base.byte().expand(1, n, 4).contiguous()
    rhs16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw)
    out16 = torch.empty_like(out)
    deep_gemm.m_grouped_gemm_w4a16_nopad(
        torch.eye(k, device='cuda', dtype=torch.bfloat16), rhs16, out16, indices)
    torch.testing.assert_close(out16, decode_weight(packed[0], raw[0]).T.bfloat16(), rtol=0, atol=0)
    print('PASS one-hot K basis, scale boundaries and nonuniform row scales')


def test_fused_k32_basis():
    # Exercise both K64 weight slices, N tail, and multiple M64 blocks.
    # Vary scales along N as well as K so a misplaced scale tile cannot pass.
    n, k = 320, 256
    codes = ((torch.arange(n, device='cuda')[:, None]*3 +
              torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).contiguous()
    raw = ((torch.arange(n, device='cuda')[:, None] +
            7*torch.arange(k//32, device='cuda')[None, :]) % 27 + 109).byte().unsqueeze(0).contiguous()
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert rhs[2] == .03125
    config = (deep_gemm.get_num_sms(), 64, 256, 128, 64, 64, 128, 2, 1)
    for m in (117, 128, 131):
        aq = torch.eye(k, device='cuda')[:m].to(torch.float8_e4m3fn)
        scales = torch.linspace(.5, 1.5, m, device='cuda').reshape(m, 1)
        routes = torch.zeros(m, 1, device='cuda', dtype=torch.int32)
        _, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
            aq, rhs[0], routes, config=config)
        out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            (aq, scales * rhs[2]), rhs[:2], out, rows, experts, tokens, aligned, config)
        expected = (decode_weight(packed[0], raw[0]).T[:m]*scales).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print('PASS fused K32 basis, shifted full scale window, M64 block boundaries and N tail')


def test_masked_shifted_scale():
    groups, n, k, padding_m = 2, 64, 128, 128
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.full((groups, n, k // 32), 124, device='cuda', dtype=torch.uint8)
    raw[0, 0, 0] = 109
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert rhs[2] == .03125
    aq = torch.eye(k, device='cuda')[:padding_m].to(torch.float8_e4m3fn).unsqueeze(0).expand(groups, -1, -1).contiguous()
    row_scales = torch.ones(groups, padding_m, 1, device='cuda', dtype=torch.float32)
    rows = torch.tensor([3, 5], device='cuda', dtype=torch.int32)
    out = torch.empty(groups, padding_m, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked((aq, row_scales * rhs[2]), rhs[:2], out, rows, 4)
    for expert, count in enumerate((3, 5)):
        expected = decode_weight(packed[expert], raw[expert]).T[:count].bfloat16()
        torch.testing.assert_close(out[expert, :count], expected, rtol=0, atol=0)
    print('PASS masked shifted scale and per-expert rows')


def _w4a8_nopad_problem(counts, n, k, with_layout=False):
    groups = len(counts)
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    scales = torch.randint(119, 133, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scales)
    quantized = quant_activation(torch.randn(sum(counts), k, device='cuda', dtype=torch.bfloat16))
    lhs = quantized[0], quantized[1] * rhs[2]
    rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
    indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
    output = torch.full((sum(counts), n), float('nan'), device='cuda', dtype=torch.bfloat16)

    def reference(weight=packed, weight_scales=scales):
        result = torch.empty_like(output)
        begin = 0
        for expert, count in enumerate(counts):
            if count:
                a = lhs[0][begin:begin + count].float() * (lhs[1][begin:begin + count] / rhs[2])
                result[begin:begin + count] = (a @ decode_weight(weight[expert], weight_scales[expert]).T).bfloat16()
            begin += count
        return result

    def call():
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], output, indices, rows)

    result = lhs, rhs, output, call, reference, packed, scales
    return result + (rows, indices) if with_layout else result


def _streamed_fused_problem(groups, tokens, topk, n, k):
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.randint(119, 133, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert rhs[2] == 1.
    lhs = quant_activation(torch.randn(tokens, k, device='cuda', dtype=torch.bfloat16))
    routes = torch.rand(tokens, groups - 1, device='cuda').topk(topk, dim=1, sorted=False).indices.int()
    config, rows, experts, sorted_tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
    assert len(config) == 9
    output = torch.empty(tokens * topk, n, device='cuda', dtype=torch.bfloat16)
    def call():
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            lhs, rhs[:2], output, rows, experts, sorted_tokens, aligned, config)
    def reference(weights=packed, scales=raw, routing=routes):
        source = routing.flatten().argsort(stable=True) // topk
        counts = torch.bincount(routing.flatten().long(), minlength=groups).cpu().tolist()
        expected = torch.empty_like(output)
        begin = 0
        for expert, count in enumerate(counts):
            if count:
                ids = source[begin:begin + count]
                a = lhs[0].float()[ids] * lhs[1][ids]
                expected[begin:begin + count] = (a @ decode_weight(weights[expert], scales[expert]).T).bfloat16()
            begin += count
        return expected
    def reroute(routing):
        new = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routing)
        assert new[0] == config
        for target, value in zip((rows, experts, sorted_tokens, aligned), new[1:5]):
            assert target.shape == value.shape
            target.copy_(value)
    return lhs, rhs, output, call, reference, packed, raw, routes, reroute


def test_routing_layout_block_m():
    # Cover both align algorithms and non-power-of-two routing blocks.
    for tokens in (1733, 2049):
        groups, topk = 97, 8
        a = torch.empty(tokens, 1024, device='cuda', dtype=torch.float8_e4m3fn)
        b = torch.empty(groups, 1024, 512, device='cuda', dtype=torch.uint8)
        routes = (torch.arange(tokens * topk, device='cuda').reshape(tokens, topk) % groups).int()
        source = routes.flatten().argsort(stable=True) // topk
        for bm in (64, 80, 96, 112, 128, 144, 160, 176, 192, 256):
            config = ((deep_gemm.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1) if bm == 256 else
                      (deep_gemm.get_num_sms(), bm, 256, 128, bm, 32 if bm > 64 else 64, 128, 2, 1))
            selected, rows, _, sorted_tokens, _, _, _ = deep_gemm.moe_align_block_size(a, b, routes, config=config)
            assert selected == config and sorted_tokens.size(1) == bm
            table = sorted_tokens.flatten()
            begin = offset = 0
            for count in rows.cpu().tolist():
                assert torch.equal(table[offset:offset + count], source[begin:begin + count].int())
                padded = (count + bm - 1) // bm * bm
                assert (table[offset + count:offset + padded] == tokens).all()
                begin += count
                offset += padded
    print('PASS routing padding blocks, expert token offsets and tail sentinels')


def _check_native_m_tile(block_m, stages, k=768):
    # Cover intermediate M tiles, output guards and inverse scales.
    sms = deep_gemm.get_num_sms()
    config = (sms, block_m, 256, 128, block_m, 32, 128, stages, 1)

    def check(call, output, expected, label, scale):
        output.fill_(float('nan'))
        call(.5)
        assert calc_diff(output, expected * .5) < .003, (label, config)
        call(1.)
        assert calc_diff(output, expected) < .003, (label, config)
        if stages == 2:
            graph = torch.cuda.CUDAGraph()
            with capture_after_warmup(graph, lambda: call(.5)):
                call(.5)
            scale.neg_()
            try:
                graph.replay()
                assert calc_diff(output, -expected * .5) < .003, (label, 'capture')
            finally:
                scale.neg_()
        print('PASS native M tile', label, config, flush=True)

    row_cases = [[0, 1, block_m - 1, block_m, block_m + 1,
                  2 * block_m - 1, 2 * block_m, 2 * block_m + 1, 3 * block_m + 1]]
    if stages == 2:
        row_cases.append([0] + [2 * block_m + 1] * 127)
    for counts in row_cases:
        n = 512 if len(counts) >= 128 else 256
        lhs, rhs, _, _, reference, _, _, rows, indices = _w4a8_nopad_problem(counts, n, k, True)
        expected = reference()
        backing = torch.full((sum(counts) + 16, n), -123., device='cuda', dtype=torch.bfloat16)
        output = backing[:-16]
        def call(inverse):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                lhs, rhs[:2], output, indices, rows, config, inv_scale=inverse)
        check(call, output, expected, 'nopad-E' + str(len(counts)), lhs[1])
        assert (backing[-16:] == -123.).all(), 'nopad output guard'

    lhs, rhs, _, _, reference, _, _, routes, _ = _streamed_fused_problem(8, 417, 4, 512, k)
    expected = reference()
    backing = torch.full((417 * 4 + 16, 512), -123., device='cuda', dtype=torch.bfloat16)
    output = backing[:-16]
    selected, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
        lhs[0], rhs[0], routes, config=config)
    assert selected == config and tokens.size(1) == block_m
    def call(inverse):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            lhs, rhs[:2], output, rows, experts, tokens, aligned, config, inv_scale=inverse)
    check(call, output, expected, 'fused', lhs[1])
    assert (backing[-16:] == -123.).all(), 'fused output guard'


def test_m96_tile():
    for stages in (2, 3, 4):
        _check_native_m_tile(96, stages)


def test_intermediate_m_tiles():
    for block_m in (80, 112, 128, 144):
        for stages in (2, 4):
            _check_native_m_tile(block_m, stages, 512 if stages == 4 else 768)


def test_wide_default_tiles():
    # Check 90% boundaries, multi-block budgets and long-K Fused M160.
    groups, topk, n = 128, 16, 256
    cases = ((65, 80), (72, 80), (73, 96), (86, 96), (87, 112),
             (100, 112), (101, 128), (115, 128), (116, 144),
             (129, 144), (130, 160), (144, 160), (145, 176), (158, 176),
             (159, 96), (172, 96), (173, 112), (176, 112), (177, 112),
             (192, 112), (256, 144), (288, 160), (289, 112),
             (320, 128), (384, 144), (512, 144), (1024, 144))
    for k in (768, 3584):
        b = torch.empty((groups, n, k // 2), dtype=torch.uint8, device='cuda')
        for em, bm in cases:
            tokens = groups * em // topk
            a = torch.empty((tokens, k), dtype=torch.float8_e4m3fn, device='cuda')
            routes = (torch.arange(tokens * topk, device='cuda').reshape(tokens, topk) % groups).int()
            config, rows, _, sorted_tokens, _, _, _ = deep_gemm.moe_align_block_size(a, b, routes)
            expected_bm = 160 if k > 2048 and bm == 176 else bm
            stages = 3 if expected_bm <= 144 else 2
            assert config == (deep_gemm.get_num_sms(), expected_bm, 256, 128,
                              expected_bm, 32, 128, stages, 1), (em, k, config)
            assert sorted_tokens.size(1) == expected_bm and bool((rows == em).all())
    for em, k in ((65, 768), (87, 768), (145, 768), (173, 3584)):
        lhs, rhs, _, _, reference, _, _, rows, indices = _w4a8_nopad_problem(
            [0, 2 * em] + [em] * 126, n, k, True)
        expected = reference()
        backing = torch.full((groups * em + 16, n), -123., dtype=torch.bfloat16, device='cuda')
        output = backing[:-16]
        for inverse in (1., .5):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                lhs, rhs[:2], output, indices, rows, inv_scale=inverse)
            assert calc_diff(output, expected * inverse) < .003
            assert bool((backing[-16:] == -123.).all())
    print('PASS automatic wide M tiles and pipeline stages')


def test_large_native_defaults():
    # Retain streamed priority and use wide native tiles when streamed is unavailable.
    groups, topk = 128, 16
    for em, n, k, bm in ((512, 256, 768, 144), (1023, 1024, 1024, 144),
                          (1024, 1024, 1024, 256)):
        tokens = groups * em // topk
        a = torch.empty((tokens, k), dtype=torch.float8_e4m3fn, device='cuda')
        b = torch.empty((groups, n, k // 2), dtype=torch.uint8, device='cuda')
        routes = (torch.arange(tokens * topk, device='cuda').reshape(tokens, topk) % groups).int()
        config, _, _, _, _, _, _ = deep_gemm.moe_align_block_size(a, b, routes)
        assert config[1] == bm, (em, n, k, config)
    for em in (320, 512, 1024):
        lhs, rhs, _, _, reference, _, _, rows, indices = _w4a8_nopad_problem(
            [0, 2 * em] + [em] * 15, 1024, 768, True)
        expected = reference()
        automatic = torch.empty_like(expected)
        explicit = torch.empty_like(expected)
        bm = 128 if em == 320 else 144
        config = (deep_gemm.get_num_sms(), bm, 256, 128, bm, 32, 128, 2, 1)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], automatic, indices, rows)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], explicit, indices, rows, configs=config)
        assert torch.equal(automatic, explicit) and calc_diff(automatic, expected) < .003
    print('PASS large native fallback and streamed priority')


def test_wide_tile_admission():
    # Small expert sets use one task-wave gate for every automatic wide tile.
    sms = deep_gemm.get_num_sms()
    for groups, em, n in ((17, 128, 512), (17, 128, 768),
                          (17, 128, 1024), (17, 128, 6144),
                          (64, 176, 512), (64, 176, 768)):
        bm = 144 if em == 128 else 192
        tasks = ((groups * em + bm - 1) // bm) * (n // 256)
        if tasks < sms:
            bm = 64
        config = (sms, bm, 256, 128, bm, 64 if bm == 64 else 32, 128, 2, 1)
        lhs, rhs, _, _, reference, _, _, rows, indices = _w4a8_nopad_problem(
            [0, 2 * em] + [em] * (groups - 2), n, 1024, True)
        expected = reference()
        backing = torch.full((groups * em + 16, n), -123., device='cuda', dtype=torch.bfloat16)
        automatic, explicit = backing[:-16], torch.empty_like(expected)
        for inverse in (1., .5):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                lhs, rhs[:2], automatic, indices, rows, inv_scale=inverse)
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                lhs, rhs[:2], explicit, indices, rows, configs=config, inv_scale=inverse)
            assert torch.equal(automatic, explicit) and calc_diff(automatic, expected * inverse) < .003
            assert bool((backing[-16:] == -123.).all())
        print(f'PASS unified wide admission {groups=} {em=} {n=} {bm=}')
    for groups, k in ((8, 1024), (17, 1024), (17, 3584)):
        em, n, topk = 128, 1024, 4
        lhs, rhs, _, _, reference, _, _, routes, _ = _streamed_fused_problem(
            groups, groups * em // topk, topk, n, k)
        config, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
            lhs[0], rhs[0], routes)
        tasks = ((groups * em + 143) // 144) * (n // 256)
        bm = 144 if tasks >= sms else 64
        assert config == (sms, bm, 256, 128, bm, 32 if bm == 144 else 64,
                          128, 3 if bm == 144 else 2, 1), (groups, k, config)
        expected = reference()
        backing = torch.full((groups * em + 16, n), -123., device='cuda', dtype=torch.bfloat16)
        for inverse in (1., .5):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                lhs, rhs[:2], backing[:-16], rows, experts, tokens, aligned, config, inv_scale=inverse)
            assert calc_diff(backing[:-16], expected * inverse) < .003
            assert bool((backing[-16:] == -123.).all())
        print(f'PASS fused wide admission {groups=} {k=} {bm=}')


def _capture_prepared(call, output, rows=None, inverse=.3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    output.fill_(float('nan'))
    with torch.cuda.stream(stream):
        deep_gemm.set_compile_mode(1)
        try:
            call(inverse)
        finally:
            deep_gemm.set_compile_mode(0)
    stream.synchronize()
    assert torch.isnan(output).all()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            call(inverse)
    except RuntimeError as error:
        assert 'must be warmed outside graph capture' in str(error)
    else:
        raise AssertionError('compile-only admitted an unprepared caller')
    live_rows = rows.clone() if rows is not None else None
    if rows is not None:
        rows.zero_()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(inverse)
    stream.synchronize()
    if rows is not None:
        assert torch.isnan(output).all()
        rows.copy_(live_rows)
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            call(1. if inverse != 1. else .3)
    except RuntimeError as error:
        assert 'must be warmed outside graph capture' in str(error)
    else:
        raise AssertionError('capture admitted an unwarmed inverse-scale variant')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(inverse)
    graph.replay()
    torch.cuda.synchronize()
    return graph, stream


def test_execution_paths():
    # FullK, rolling and streamed kernels prepare only the requested scale variant.
    for mode in ('full', 'rolling', 'streamed'):
        streamed = mode == 'streamed'
        for fused in (False, True):
            groups, n, k = 8, 1344 if streamed else 512, 1024 if streamed else 768
            config = ((deep_gemm.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1) if streamed else
                      (deep_gemm.get_num_sms(), 64, 256, 128, 64, 64, 128, 2, 1) if mode == 'full' else
                      (deep_gemm.get_num_sms(), 160, 256, 128, 160, 32, 128, 3, 1))
            if fused:
                lhs, rhs, output, _, reference, packed, raw, routes, _ = _streamed_fused_problem(
                    groups, 193 if streamed else 417, 4, n, k)
                selected, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
                    lhs[0], rhs[0], routes, config=config)
                assert selected == config and tokens.size(1) == config[1]
                def call(inverse):
                    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                        lhs, rhs[:2], output, rows, experts, tokens, aligned, config, inv_scale=inverse)
            else:
                counts = [0, 1, 17, 65, 256, 257, 128, 0] if streamed else [0, 1, 31, 32, 65, 160, 161, 257]
                lhs, rhs, output, _, reference, packed, raw, rows, indices = _w4a8_nopad_problem(
                    counts, n, k, True)
                def call(inverse):
                    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                        lhs, rhs[:2], output, indices, rows, config, inv_scale=inverse)
            graph, stream = _capture_prepared(call, output, None if fused else rows)
            assert calc_diff(output, reference() * .3) < .003
            packed.bitwise_xor_(0x88)
            raw.copy_(raw.flip(-1))
            changed = deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
            assert changed[2] == rhs[2]
            rhs[0].copy_(changed[0])
            rhs[1].copy_(changed[1])
            lhs[1].mul_(-.5)
            if fused:
                routes.add_(1).remainder_(groups)
                updated = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes, config=config)
                assert updated[0] == config
                for target, value in zip((rows, experts, tokens, aligned), updated[1:5]):
                    assert target.shape == value.shape
                    target.copy_(value)
            else:
                counts[:] = counts[1:] + counts[:1]
                rows.copy_(torch.tensor(counts, device='cuda', dtype=torch.int32))
                indices.copy_(torch.repeat_interleave(
                    torch.arange(groups, device='cuda', dtype=torch.int32), rows.long()))
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(output).all()
            assert calc_diff(output, reference() * .3) < .003
            with torch.cuda.stream(stream):
                call(.3)
                consumed = output.clone()
            stream.synchronize()
            assert torch.equal(consumed, output)
            unit_graph, _ = _capture_prepared(call, output, None if fused else rows, inverse=1.)
            assert calc_diff(output, reference()) < .003
            graph.replay()
            torch.cuda.synchronize()
            assert calc_diff(output, reference() * .3) < .003
            # A different non-unit scalar reuses the warmed non-unit variant.
            nonunit_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(nonunit_graph, stream=stream):
                call(.7)
            nonunit_graph.replay()
            torch.cuda.synchronize()
            assert calc_diff(output, reference() * .7) < .003
            unit_graph.replay()
            torch.cuda.synchronize()
            assert calc_diff(output, reference()) < .003
            print(f'PASS execution {mode=} {fused=}: variant-specific warmup, capture and live inputs')


def test_explicit_execution_configs():
    # Explicit tiles determine execution without hidden default overrides.
    sms = deep_gemm.get_num_sms()
    lhs, rhs, out, call, reference, _, _, rows, indices = _w4a8_nopad_problem([0, 1, 17, 65], 256, 768, True)
    expected = reference()
    tile = (sms, 32, 256, 128, 32, 64, 128, 2, 1)
    narrow_warp_tile = tile[:5] + (32,) + tile[6:]
    wide_tile = (sms, 64, 256, 128, 64, 64, 128, 2, 1)
    for config in (tile, narrow_warp_tile, wide_tile):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows, config, inv_scale=.3)
        assert calc_diff(out, expected * .3) < .003, (config, calc_diff(out, expected * .3))
    oversized = (sms, 64, 1 << 30, 128, 64, 64, 128, 2, 4)
    multi_m_warp = (sms, 128, 256, 128, 64, 64, 128, 2, 1)
    split_k_tile = (sms, 16, 256, 128, 16, 64, 64, 2, 1)
    legacy_m128 = (sms, 128, 128, 128, 128, 32, 128, 2, 1)
    for bad in (oversized, multi_m_warp, split_k_tile, legacy_m128):
        try:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows, bad)
        except (RuntimeError, TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted invalid tile config {bad}')
    try:
        deep_gemm.moe_align_block_size(lhs[0], rhs[0], indices[:, None], config=multi_m_warp)
    except RuntimeError as error:
        assert 'block_m == warp_m' in str(error)
    else:
        raise AssertionError('alignment accepted a multi-M-warp native tile')
    print('PASS explicit native tiles and invalid configurations')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    utils.set_acc_check(True)
    utils.set_benchmark(False)

    if is_ppu1v5_device():
        test_execution_paths()
        test_explicit_execution_configs()
        test_routing_layout_block_m()
        test_m96_tile()
        test_intermediate_m_tiles()
        test_wide_tile_admission()
        test_wide_default_tiles()
        test_large_native_defaults()
        test_grouped_gemm()
        test_inv_scale()
        test_nopad_expert_limit()
        test_nopad_n_expand()
        test_nopad_resident_a_and_scales()
        test_fused_nopad_equivalence()
        test_boundaries()
        test_tensor_scale_shift()
        test_preprocessing_layout()
        test_basis()
        test_fused_k32_basis()
        test_masked_shifted_scale()
    else:
        print('Skipping W4FA8: requires PPU1.5')
