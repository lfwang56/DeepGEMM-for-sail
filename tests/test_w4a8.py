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
    # Cover uncached scales, resident A with cached scales, and K-warp reduction.
    for k, config in [(128, (sms, 16, 256, 128, 16, 64, 128, 2, 1)),
                      (384, (sms, 64, 256, 128, 64, 64, 128, 2, 3)),
                      (256, (sms, 32, 128, 128, 32, 64, 64, 2, 1))]:
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

    k, n, capacity = 256, 192, 67
    packed = torch.randint(256, (4, n, k//2), device='cuda', dtype=torch.uint8)
    raw = torch.full((4, n, k//32), 127, device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    lhs = quant_activation(torch.randn(4, capacity, k, device='cuda', dtype=torch.bfloat16))
    rows = torch.tensor([0, 1, 33, 67], device='cuda', dtype=torch.int32)
    def masked(a, output, **kwargs):
        config = (sms, 32, 128, 128, 32, 64, 128, 2, 1)
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
            configs.append((deep_gemm.get_num_sms(), 128, 256, 128, 64, 64, 128, 2, 3))
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
    # Exercise both K64 weight slices, N tail, and multiple compact M64 blocks.
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





def test_tuned_nopad_dispatch():
    # Cover all three small-M kernels, uneven/empty experts, explicit native
    # configs, optional row counts and non-default output scaling.
    for groups, n, k in ((28, 6144, 3584), (56, 3584, 3072)):
        packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
        raw = torch.full((groups, n, k // 32), 127, device='cuda', dtype=torch.uint8)
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        for mean_rows in (1, 17, 33):
            counts = [mean_rows] * groups
            counts[0] += counts[-1]
            counts[-1] = 0
            rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
            indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
            quantized = quant_activation(torch.randn(sum(counts), k, device='cuda', dtype=torch.bfloat16))
            lhs = quantized[0], quantized[1] * rhs[2]
            output = torch.full((sum(counts), n), float('nan'), device='cuda', dtype=torch.bfloat16)
            reference = torch.empty_like(output)
            begin = 0
            for expert, count in enumerate(counts):
                if count:
                    a = lhs[0][begin:begin + count].float() * quantized[1][begin:begin + count]
                    reference[begin:begin + count] = (a @ decode_weight(packed[expert], raw[expert]).T).bfloat16()
                begin += count
            call = deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad
            call(lhs, rhs[:2], output, indices, rows)
            assert torch.isfinite(output).all()
            assert calc_diff(output, reference) < .0015
            baseline = output.clone()
            graph = torch.cuda.CUDAGraph()
            with capture_after_warmup(graph, lambda: call(lhs, rhs[:2], output, indices, rows)):
                call(lhs, rhs[:2], output, indices, rows)
            lhs[1].neg_()
            graph.replay()
            assert torch.equal(output, -baseline)
            lhs[1].neg_()
            call(lhs, rhs[:2], output, indices, rows, inv_scale=.25)
            assert calc_diff(output, reference * .25) < .0015
            call(lhs, rhs[:2], output, indices)
            assert calc_diff(output, reference) < .0015
            configs = (deep_gemm.get_num_sms(), 16, 256, 128, 16, 64, 128, 2, 1)
            call(lhs, rhs[:2], output, indices, rows, configs)
            assert calc_diff(output, reference) < .0015
        del packed, raw, rhs, lhs, output, reference, baseline, graph
        torch.cuda.empty_cache()
    print('PASS native W4A8 small-M dispatch, configs, scales and Graph inputs')


def test_tuned_fused_dispatch():
    # Constant-K E2M1 codes give a reference independent of the optimized MMA.
    groups, n, k, tokens, topk = 896, 768, 3584, 8192, 16
    codes = (torch.arange(groups, device='cuda')[:, None] + torch.arange(n, device='cuda')[None, :]) % 16
    packed = (codes.to(torch.uint8) * 17)[:, :, None].expand(groups, n, k // 2).contiguous()
    raw = torch.full((groups, n, k // 32), 127, device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device='cuda')
    weight = values[codes]
    quantized = quant_activation(torch.randn(tokens, k, device='cuda', dtype=torch.bfloat16))
    lhs = quantized[0], quantized[1] * rhs[2]
    for balanced in (True, False):
        routes = ((torch.arange(tokens * topk, device='cuda').reshape(tokens, topk) % groups).int()
                  if balanced else torch.rand(tokens, groups, device='cuda').topk(topk, dim=1).indices.int())
        configs, rows, experts, sorted_tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
        order = routes.flatten().argsort(stable=True)
        source = order // topk
        counts = torch.bincount(routes.flatten().long(), minlength=groups).cpu().tolist()
        sums = (lhs[0].float().sum(1) * quantized[1][:, 0])[source]
        reference = torch.empty(tokens * topk, n, device='cuda', dtype=torch.bfloat16)
        begin = 0
        for expert, count in enumerate(counts):
            reference[begin:begin + count] = (sums[begin:begin + count, None] * weight[expert]).bfloat16()
            begin += count
        output = torch.full_like(reference, float('nan'))
        def call():
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                lhs, rhs[:2], output, rows, experts, sorted_tokens, aligned, configs)
        call()
        assert torch.isfinite(output).all()
        assert calc_diff(output, reference) < .003
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, lambda: call()):
            call()
        lhs[1].neg_()
        graph.replay()
        assert calc_diff(output, -reference) < .003
        lhs[1].zero_()
        graph.replay()
        assert torch.equal(output, torch.zeros_like(output))
        # Restore a fresh activation scale for the other routing distribution.
        quantized = quant_activation(torch.randn(tokens, k, device='cuda', dtype=torch.bfloat16))
        lhs = quantized[0], quantized[1] * rhs[2]
    print('PASS native W4A8 routed dispatch, balanced/random routes and Graph inputs')


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


def test_compact_fused():
    for groups, tokens, topk, n, k in ((13, 29, 3, 320, 768), (31, 191, 4, 512, 1024),
                                      (64, 769, 4, 1024, 512), (97, 769, 4, 512, 768),
                                      (129, 1021, 4, 768, 1024)):
        lhs, _, output, call, reference, *_ = _streamed_fused_problem(groups, tokens, topk, n, k)
        expected = reference()
        call()
        assert calc_diff(output, expected) < .003
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, call):
            call()
        lhs[1].neg_()
        graph.replay()
        torch.cuda.synchronize()
        assert calc_diff(output, -expected) < .003


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


def test_streamed_fused_replay_updates():
    for groups, n, k, mean_rows in ((10, 2048, 3584, 512), (10, 2048, 2048, 1024),
                                   (17, 1536, 1024, 1024)):
        lhs, rhs, output, call, reference, packed, raw, routes, reroute = _streamed_fused_problem(
            groups, groups * mean_rows // 4, 4, n, k)
        call()
        assert calc_diff(output, reference()) < .003
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, call) as stream:
            call()
        lhs[0].copy_(lhs[0].float().roll(1, 0).to(lhs[0].dtype))
        lhs[1].neg_()
        graph.replay()
        torch.cuda.synchronize()
        assert calc_diff(output, reference()) < .003
        weights, scales = packed.bitwise_xor(0x88), raw.flip(-1).contiguous()
        updated = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, weights, scales)
        rhs[0].copy_(updated[0])
        rhs[1].copy_(updated[1])
        routing = (routes + 1) % groups
        reroute(routing)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(weights, scales, routing)
        assert calc_diff(output, expected) < .003
        with torch.cuda.stream(stream):
            call()
            consumed = output.clone()
        stream.synchronize()
        assert calc_diff(consumed, expected) < .003


def test_native_strategy_domains():
    # Original profiles used E28/E56 and two projection shapes. These cases
    # exercise scheduler bounds, N tails, and a reduced configured CU count.
    sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(min(sms, 20))
        for groups, mean_rows, n, k in ((13, 1, 320, 768), (31, 23, 512, 1024),
                                       (64, 49, 1024, 512), (97, 137, 512, 768)):
            counts = [mean_rows] * groups
            counts[0] += counts[-1]
            counts[-1] = 0
            _, _, output, call, reference, _, _ = _w4a8_nopad_problem(counts, n, k)
            call()  # The first M64/routed call must also initialize dynamic SMEM.
            assert torch.isfinite(output).all(), (groups, mean_rows, n, k)
            assert calc_diff(output, reference()) < .003
    finally:
        deep_gemm.set_num_sms(sms)


def test_streamed_nopad_replay_updates():
    # Full-width streamed shapes outside the original profiles. Include an
    # expert that is empty, a one-row expert, unequal chunks and nonuniform
    # UE8M0 scales across both N and K. All oracle work stays outside capture.
    for groups, n, k in ((10, 2048, 2048), (17, 1536, 1024)):
        counts = [1024] * groups
        counts[0], counts[1], counts[-1] = 0, 1, 3071
        lhs, rhs, output, call, reference, packed, scales = _w4a8_nopad_problem(counts, n, k)
        call()
        assert calc_diff(output, reference()) < .003
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, call) as stream:
            call()
        graph.replay()
        torch.cuda.synchronize()
        assert calc_diff(output, reference()) < .003
        lhs[1].mul_(-.5)
        graph.replay()
        torch.cuda.synchronize()
        assert calc_diff(output, reference()) < .003
        # Update both packed weights and physical scales at their captured
        # addresses. A cached decoded-weight tile would fail this comparison.
        changed_packed = packed.bitwise_xor(0x88)
        changed_scales = scales.flip(-1).contiguous()
        changed_rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, changed_packed, changed_scales)
        assert changed_rhs[2] == rhs[2]
        rhs[0].copy_(changed_rhs[0])
        rhs[1].copy_(changed_rhs[1])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert calc_diff(output, reference(changed_packed, changed_scales)) < .003
        # Caller work can consume the output immediately after the API joins.
        with torch.cuda.stream(stream):
            call()
            consumed = output.clone()
        stream.synchronize()
        assert torch.equal(consumed, output)


def _check_routed_fused_metadata_layout(n, k, tokens, groups=97, topk=8):
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.randint(119, 133, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    quantized = quant_activation(torch.randn(tokens, k, device='cuda', dtype=torch.bfloat16))
    lhs = quantized[0], quantized[1] * rhs[2]
    # Active experts have about 144 rows: BlockM32 pads to 160 and BlockM64
    # pads to 192. Leave the final expert empty and preserve uneven counts.
    routes = ((torch.arange(tokens, device='cuda')[:, None] * 7 +
               torch.arange(topk, device='cuda')[None, :]) % (groups - 1)).int()
    order = routes.flatten().argsort(stable=True)
    source = order // topk
    counts = torch.bincount(routes.flatten().long(), minlength=groups).cpu().tolist()
    output = torch.empty(tokens * topk, n, device='cuda', dtype=torch.bfloat16)
    reference = torch.empty_like(output)
    begin = 0
    for expert, count in enumerate(counts):
        if count:
            ids = source[begin:begin + count]
            a = lhs[0].float()[ids] * quantized[1][ids]
            reference[begin:begin + count] = (a @ decode_weight(packed[expert], raw[expert]).T).bfloat16()
        begin += count
    for block_m in (64, 32):
        # Construct the corresponding token layout rather than changing only
        # the GEMM config after moe_align_block_size has produced its metadata.
        config = (deep_gemm.get_num_sms(), block_m, 256, 128, block_m, 64, 128, 2, 1)
        selected, rows, experts, sorted_tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
            lhs[0], rhs[0], routes, config=config)
        assert selected == config
        def call():
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                lhs, rhs[:2], output, rows, experts, sorted_tokens, aligned, config)
        call()
        assert calc_diff(output, reference) < .003
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, lambda: call()):
            call()
        lhs[1].neg_()
        graph.replay()
        assert calc_diff(output, -reference) < .003
        lhs[1].neg_()


def test_routed_fused_metadata_layout():
    _check_routed_fused_metadata_layout(512, 768, 1733)
    # Exercise large K with random packed values and scales that vary across
    # K32 groups, independently of the constant-K K3 oracle.
    _check_routed_fused_metadata_layout(768, 3584, 2048)


def test_routed_fused_small_expert_set():
    _check_routed_fused_metadata_layout(6144, 3584, 448, groups=28)
    # Two internal M blocks per expert exercise interleaving of full/tail pairs.
    _check_routed_fused_metadata_layout(2048, 768, 896, groups=28)


def test_fused_large_output_offset():
    # A legal routed problem crosses the signed 32-bit element offset at expert
    # 256. Keep the oracle small while exercising the real public GPU kernel.
    groups, r, n, k = 512, 256, 32768, 512
    packed = torch.full((groups, n, k // 2), 0x22, device='cuda', dtype=torch.uint8)
    raw = torch.full((groups, n, k // 32), 127, device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    a = torch.ones(groups * r, k, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.ones(groups * r, 1, device='cuda') * rhs[2]
    routes = torch.arange(groups, device='cuda').repeat_interleave(r).int().view(-1, 1)
    config, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(a, rhs[0], routes)
    out = torch.empty(groups * r, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
        (a, scales), rhs[:2], out, rows, experts, tokens, aligned, config)
    probe_rows = torch.tensor([0, 65535, 65536, 65696, groups * r - 1], device='cuda')
    selected = out[probe_rows][:, [0, 255, n - 1]]
    torch.testing.assert_close(selected, torch.full_like(selected, float(k)), rtol=0, atol=0)


def test_streamed_capture_prepared_plans():
    # Warm different shapes on one caller, then capture each derived plan.
    # A complete ordinary warmup must leave no cold compiler work in capture.
    from pathlib import Path
    import os
    stream = torch.cuda.Stream()
    problems = [_streamed_fused_problem(10, t, 4, 2048, 2048) for t in (1280, 2560, 10240)]
    stream.wait_stream(torch.cuda.current_stream())
    for problem in problems:
        with torch.cuda.stream(stream):
            problem[3]()
    stream.synchronize()
    cache = Path(os.environ.get('DG_JIT_CACHE_DIR', str(Path.home() / '.deep_gemm')))
    before = {str(p) for p in cache.rglob('kernel.hgbin')}
    for problem in reversed(problems):
        output, call, reference = problem[2:5]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        graph.replay()
        torch.cuda.synchronize()
        assert calc_diff(output, reference()) < .003
    assert before == {str(p) for p in cache.rglob('kernel.hgbin')}


def test_compact_intermediate_rows_and_skew():
    for groups, r, n, k in ((17, 65, 1280, 768), (53, 95, 1792, 1536),
                             (131, 257, 2560, 896), (257, 511, 512, 640)):
        for active in (groups, max(1, groups // 8)):
            counts = [0] * groups
            for i in range(active):
                counts[i] = groups * r // active + (i < groups * r % active)
            lhs, rhs, output, nopad, reference, packed, scales, rows, indices = _w4a8_nopad_problem(
                counts, n, k, with_layout=True)
            config, frows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
                lhs[0], rhs[0], indices.view(-1, 1))
            fused_output = torch.empty_like(output)
            def fused():
                deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                    lhs, rhs[:2], fused_output, frows, experts, tokens, aligned, config)
            nopad()
            fused()
            expected = reference()
            assert calc_diff(output, expected) < .003
            assert calc_diff(fused_output, expected) < .003
            graph = torch.cuda.CUDAGraph()
            with capture_after_warmup(graph, lambda: (nopad(), fused())):
                nopad()
                fused()
            # Change the distribution itself after capture: balanced <-> hot,
            # with moving empty experts. Tile costs must use fresh row values.
            updated_active = max(1, groups // 8) if active == groups else groups
            counts[:] = [0] * groups
            for i in range(updated_active):
                counts[(i + 1) % groups] = groups * r // updated_active + (i < groups * r % updated_active)
            rows.copy_(torch.tensor(counts, device='cuda', dtype=torch.int32))
            indices.copy_(torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long()))
            changed = deep_gemm.moe_align_block_size(lhs[0], rhs[0], indices.view(-1, 1))
            assert changed[0] == config
            for target, value in zip((frows, experts, tokens, aligned), changed[1:5]):
                assert target.shape == value.shape
                target.copy_(value)
            new_packed = packed.bitwise_xor(0x88)
            new_rhs = deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, new_packed, scales)
            assert new_rhs[2] == rhs[2]
            rhs[0].copy_(new_rhs[0])
            rhs[1].copy_(new_rhs[1])
            lhs[1].mul_(-.5)
            graph.replay()
            torch.cuda.synchronize()
            expected = reference(new_packed, scales)
            assert calc_diff(output, expected) < .003
            assert calc_diff(fused_output, expected) < .003


def _capture_on_fresh_stream(call):
    call()  # An ordinary call prepares resources only for this caller.
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # The new caller has its own internal streams/events and must prepare them.
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    from pathlib import Path
    import os
    cache = Path(os.environ.get('DG_JIT_CACHE_DIR', str(Path.home() / '.deep_gemm')))
    before = {str(p) for p in cache.rglob('kernel.hgbin')}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert before == {str(p) for p in cache.rglob('kernel.hgbin')}
    return graph


def test_streamed_nopad_per_caller_preparation():
    _, _, output, call, reference, _, _ = _w4a8_nopad_problem([2048] * 19, 2560, 1280)
    _capture_on_fresh_stream(call)
    assert calc_diff(output, reference()) < .003


def test_streamed_fused_per_caller_preparation():
    _, _, output, call, reference, _, _, _, _ = _streamed_fused_problem(23, 11776, 4, 2560, 1280)
    _capture_on_fresh_stream(call)
    assert calc_diff(output, reference()) < .003


def test_w4a8_cold_capture_requires_warmup():
    _, _, output, call, reference, _, _ = _w4a8_nopad_problem([70] * 5, 1088, 640)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            call()
    except RuntimeError as error:
        assert 'must be warmed outside graph capture' in str(error)
    else:
        raise AssertionError('cold capture unexpectedly loaded a W4A8 module')
    # The rejected cold request must leave the device usable.
    _capture_on_fresh_stream(call)
    assert calc_diff(output, reference()) < .003


def test_streamed_partial_n_basis_and_replay():
    # One expert exceeds512MiB and leaves an N64 tail in a partial slice.
    # Check all columns at sampled rows with an independent one-hot K oracle.
    groups, n, k, m = 8, 65600, 8192, 8192
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.randint(119, 133, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    token_a = torch.zeros(m, k, device='cuda', dtype=torch.bfloat16)
    token_a[torch.arange(m, device='cuda'), torch.arange(m, device='cuda') % k] = 1
    token_a = token_a.to(torch.float8_e4m3fn)
    token_scale = (.5 + (torch.arange(m, device='cuda') % 5).float() / 8).view(-1, 1)
    routes = (torch.arange(m, device='cuda')[torch.randperm(m, device='cuda')] % groups).int()
    source = routes.argsort(stable=True)
    lhs = token_a.float()[source].to(token_a.dtype).contiguous(), token_scale[source] * rhs[2]
    indices = routes[source].contiguous()
    rows = torch.bincount(routes.long(), minlength=groups).int()
    config, frows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
        token_a, rhs[0], routes.view(-1, 1))
    nopad_out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    fused_out = torch.empty_like(nopad_out)
    def call():
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], nopad_out, indices, rows)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            (token_a, token_scale * rhs[2]), rhs[:2], fused_out, frows, experts, tokens, aligned, config)
    probes = torch.tensor([0, 1, 1023, 1024, m // 2, m - 2, m - 1], device='cuda')
    columns = torch.arange(n, device='cuda')
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device='cuda')
    shift = 0
    def check():
        selected_tokens = source[probes]
        e, ki = routes[selected_tokens, None].long(), ((selected_tokens + shift) % k)[:, None]
        codes = (packed[e, columns, ki // 2].int() >> (4 * (ki % 2))) & 15
        expected = (values[codes] * torch.exp2(raw[e, columns, ki // 32].float() - 127) *
                    token_scale[selected_tokens]).bfloat16()
        torch.testing.assert_close(nopad_out[probes], expected, rtol=0, atol=0)
        torch.testing.assert_close(fused_out[probes], expected, rtol=0, atol=0)
    call()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    # Move empty experts and change all captured weight/activation/route values.
    routes.copy_(routes % (groups // 2) + groups // 2)
    shift = 1
    token_a.copy_(token_a.float().roll(1, 1).to(token_a.dtype))
    source = routes.argsort(stable=True)
    indices.copy_(routes[source])
    rows.copy_(torch.bincount(routes.long(), minlength=groups).int())
    lhs[0].copy_(token_a.float()[source].to(token_a.dtype))
    token_scale.neg_()
    lhs[1].copy_(token_scale[source] * rhs[2])
    changed = deep_gemm.moe_align_block_size(token_a, rhs[0], routes.view(-1, 1))
    assert changed[0] == config
    for target, value in zip((frows, experts, tokens, aligned), changed[1:5]):
        assert target.shape == value.shape
        target.copy_(value)
    packed.bitwise_xor_(0x88)
    raw.copy_(raw.flip(-1))
    changed_rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert changed_rhs[2] == rhs[2]
    rhs[0].copy_(changed_rhs[0])
    rhs[1].copy_(changed_rhs[1])
    graph.replay()
    torch.cuda.synchronize()
    check()


def test_streamed_n_tail():
    lhs, rhs, output, call, reference, _, _ = _w4a8_nopad_problem([1024] * 10, 1344, 2048)
    call()
    assert calc_diff(output, reference()) < .003
    _, _, output, call, reference, _, _, _, _ = _streamed_fused_problem(13, 3328, 4, 1344, 2048)
    call()
    assert calc_diff(output, reference()) < .003


def test_routed_fused_replay_updates():
    # Exercise regular and irregular token order through the same gather path.
    # Cover full tiles, partial tiles, concentrated routes and empty experts.
    # The same captured graph must observe each route and tensor update.
    torch.manual_seed(202611010923)
    groups, tokens, topk, n, k = 192, 3840, 8, 768, 3584
    lhs, rhs, output, call, reference, packed, scales, _, reroute = _streamed_fused_problem(
        groups, tokens, topk, n, k)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_after_warmup(graph, lambda: call()):
        call()
    slots = torch.arange(tokens * topk, device='cuda').reshape(tokens, topk)
    for active in (groups, groups - 1, 128, 160, groups // 2, 1):
        routes = (slots % active).int()
        packed.random_(0, 256)
        scales.random_(119, 133)
        weights, weight_scales, inv_scale = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scales)
        assert inv_scale == 1.
        rhs[0].copy_(weights)
        rhs[1].copy_(weight_scales)
        lhs[0].copy_(torch.randn_like(lhs[0].float()).to(lhs[0].dtype))
        lhs[1].uniform_(.25, 1.25)
        reroute(routes)
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert calc_diff(output, reference(routing=routes)) < .003



def test_native_optimized_inv_scale(k=768):
    # Switch unit/non-unit variants across row distributions to catch stale cache keys.
    torch.manual_seed(41)
    groups, n = 97, 256
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.randint(123, 132, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)[:2]
    lhs = quant_activation(torch.randn(groups * 160, k, device='cuda', dtype=torch.bfloat16))
    output = torch.empty(groups * 160, n, device='cuda', dtype=torch.bfloat16)
    reference = torch.empty_like(output)
    for skew in (False, True):
        counts = [160] * groups
        if skew:
            counts[0], counts[1] = 170, 150
        rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
        indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
        for inverse in (1., .3, 2., 1.):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                (lhs[0], lhs[1] * inverse), rhs, reference, indices, rows)
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs, output, indices, rows, inv_scale=inverse)
            assert torch.equal(output, reference), ('nopad', skew, inverse)
    print('PASS native inv_scale nopad queue variants', flush=True)
    lhs, rhs, output, _, _, _, _, routes, _ = _streamed_fused_problem(groups, 1940, 8, n, k)
    config, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
    reference = torch.empty_like(output)
    for inverse in (1., .3, 2., 1.):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            (lhs[0], lhs[1] * inverse), rhs[:2], reference, rows, experts, tokens, aligned, config)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            lhs, rhs[:2], output, rows, experts, tokens, aligned, config, inv_scale=inverse)
        assert torch.equal(output, reference), ('fused', inverse)
    print('PASS native optimized inv_scale, row distributions and cache variants')



def test_streamed_inv_scale():
    # Enough reuse for streamed dispatch; do not let non-unit scales choose native.
    torch.manual_seed(42)
    groups, n, k = 8, 1024, 1024
    lhs_f, rhs, out_f, _, _, _, _, routes, _ = _streamed_fused_problem(groups, 2048, 4, n, k)
    config, rows_f, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs_f[0], rhs[0], routes)
    lhs_n = quant_activation(torch.randn(groups * 1024, k, device='cuda', dtype=torch.bfloat16))
    rows_n = torch.full((groups,), 1024, device='cuda', dtype=torch.int32)
    indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows_n.long())
    out_n = torch.empty(groups * 1024, n, device='cuda', dtype=torch.bfloat16)
    def nopad(lhs, out, inverse):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows_n, inv_scale=inverse)
    def fused(lhs, out, inverse):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            lhs, rhs[:2], out, rows_f, experts, tokens, aligned, config, inv_scale=inverse)
    for lhs, out, call in ((lhs_n, out_n, nopad), (lhs_f, out_f, fused)):
        reference = torch.empty_like(out)
        for inverse in (1., .3, 2., 1.):
            call((lhs[0], lhs[1] * inverse), reference, 1.)
            call(lhs, out, inverse)
            assert torch.equal(out, reference), (call.__name__, inverse)
        call(lhs, out, .3)
        graph = torch.cuda.CUDAGraph()
        with capture_after_warmup(graph, lambda: call(lhs, out, 0.3)):
            call(lhs, out, .3)
        expected = out.clone()
        lhs[1].neg_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, -expected), call.__name__
    print('PASS streamed inv_scale, unit/non-unit cache keys and Graph replay')



def test_routing_layout_block_m():
    # Cover both align algorithms and non-power-of-two routing blocks.
    for tokens in (1733, 2049):
        groups, topk = 97, 8
        a = torch.empty(tokens, 1024, device='cuda', dtype=torch.float8_e4m3fn)
        b = torch.empty(groups, 1024, 512, device='cuda', dtype=torch.uint8)
        routes = (torch.arange(tokens * topk, device='cuda').reshape(tokens, topk) % groups).int()
        source = routes.flatten().argsort(stable=True) // topk
        for bm in (64, 160, 176, 192, 256):
            config = ((deep_gemm.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1) if bm == 256 else
                      (deep_gemm.get_num_sms(), bm, 256, 128, bm, 32 if bm > 128 else 64, 128, 2, 1))
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
    oversized = (sms, 128, 1 << 30, 128, 16, 64, 128, 2, 4)
    for bad in (oversized,):
        try:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows, bad)
        except (RuntimeError, TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted invalid tile config {bad}')
    # Alignment returns the same tile tuple that GEMM consumes.
    lhs_f, rhs_f, out_f, _, ref_f, _, _, routes, _ = _streamed_fused_problem(8, 2048, 4, 1024, 1024)
    core = (sms, 256, 256, 128, 64, 64, 128, 4, 1)
    config = core
    selected, counts, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
        lhs_f[0], rhs_f[0], routes, config=config)
    assert selected == config and tokens.size(1) == config[1]
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
        lhs_f, rhs_f[:2], out_f, counts, experts, tokens, aligned, config, inv_scale=.3)
    assert calc_diff(out_f, ref_f() * .3) < .003
    print('PASS explicit native/streamed execution configs and invalid combinations')

def test_native_capture_readiness():
    # Warm an empty single-tile queue, then capture nonempty work with the
    # other inverse-scale module without compilation or fallback.
    lhs, rhs, output, _, reference, _, _, rows, indices = _w4a8_nopad_problem([170] * 65, 256, 768, True)
    sms = deep_gemm.get_num_sms()
    config = (sms, 160, 256, 128, 160, 32, 128, 2, 1)
    counts = rows.clone()
    rows.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], output, indices, rows, config)
    stream.synchronize()
    rows.copy_(counts)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
            lhs, rhs[:2], output, indices, rows, config, inv_scale=.3)
    graph.replay()
    torch.cuda.synchronize()
    assert calc_diff(output, reference() * .3) < .003
    fresh = torch.cuda.Stream()
    fresh.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(fresh):
        deep_gemm.set_compile_mode(1)
        try:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], output, indices, rows, config)
        finally:
            deep_gemm.set_compile_mode(0)
    fresh.synchronize()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=fresh):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], output, indices, rows, config)
    except RuntimeError as error:
        assert 'must be warmed outside graph capture' in str(error)
    else:
        raise AssertionError('unprepared caller stream accepted native capture')
    torch.cuda.synchronize()
    # Omitted counts are constructed with a fixed-size GPU reduction in capture.
    def without_rows():
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], output, indices, configs=config, inv_scale=.3)
    graph2 = torch.cuda.CUDAGraph()
    with capture_after_warmup(graph2, without_rows):
        without_rows()
    graph2.replay()
    torch.cuda.synchronize()
    assert calc_diff(output, reference() * .3) < .003
    print('PASS native all-empty warmup, alternate inv_scale capture and per-caller readiness')

def test_streamed_complete_preparation():
    for fused in (False, True):
        _check_streamed_complete_preparation(fused)


def _check_streamed_complete_preparation(fused):
    # Compile-only prepares modules without writing output; ordinary warmup
    # establishes caller readiness. The first non-unit call is captured.
    config = (deep_gemm.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1)
    if fused:
        lhs, rhs, output, _, reference, _, _, routes, _ = _streamed_fused_problem(8, 2048, 4, 1024, 1024)
        selected, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
        assert selected == config
        def call(inverse):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
                lhs, rhs[:2], output, rows, experts, tokens, aligned, config, inv_scale=inverse)
    else:
        lhs, rhs, output, _, reference, _, _, rows, indices = _w4a8_nopad_problem([1024] * 8, 1024, 1024, True)
        def call(inverse):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                lhs, rhs[:2], output, indices, rows, config, inv_scale=inverse)
    output.fill_(float('nan'))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deep_gemm.set_compile_mode(1)
        try:
            call(1.)
        finally:
            deep_gemm.set_compile_mode(0)
    stream.synchronize()
    assert torch.isnan(output).all()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            call(.3)
    except RuntimeError as error:
        assert 'must be warmed outside graph capture' in str(error)
    else:
        raise AssertionError('compile-only incorrectly admitted capture')
    with torch.cuda.stream(stream):
        call(1.)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(.3)
    graph.replay()
    torch.cuda.synchronize()
    assert calc_diff(output, reference() * .3) < .003
    print('PASS complete streamed compile-only preparation and alternate numerical variant capture')

def test_streamed_empty_warmup():
    counts = [0, 1, 17, 65, 32, 0, 4, 11]
    lhs, rhs, output, _, reference, _, _, rows, indices = _w4a8_nopad_problem(counts, 1024, 1024, True)
    config = (deep_gemm.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1)
    live_rows = rows.clone()
    rows.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def call(inverse):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
            lhs, rhs[:2], output, indices, rows, config, inv_scale=inverse)
    with torch.cuda.stream(stream):
        call(1.)
    stream.synchronize()
    assert torch.isnan(output).all()
    rows.copy_(live_rows)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(.3)
    graph.replay()
    torch.cuda.synchronize()
    assert calc_diff(output, reference() * .3) < .003
    print('PASS streamed all-empty warmup and nonempty alternate-variant replay')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    utils.set_acc_check(True)
    utils.set_benchmark(False)

    if is_ppu1v5_device():
        test_streamed_empty_warmup()
        test_streamed_complete_preparation()
        test_native_capture_readiness()
        test_explicit_execution_configs()
        test_grouped_gemm()
        test_routing_layout_block_m()
        test_streamed_partial_n_basis_and_replay()
        test_streamed_n_tail()
        test_w4a8_cold_capture_requires_warmup()
        test_streamed_nopad_per_caller_preparation()
        test_streamed_fused_per_caller_preparation()
        test_compact_intermediate_rows_and_skew()
        test_fused_large_output_offset()
        test_streamed_capture_prepared_plans()
        test_inv_scale()
        test_native_optimized_inv_scale()
        test_streamed_inv_scale()
        test_tuned_nopad_dispatch()
        test_tuned_fused_dispatch()
        test_native_strategy_domains()
        test_compact_fused()
        test_streamed_fused_replay_updates()
        test_streamed_nopad_replay_updates()
        test_routed_fused_replay_updates()
        test_routed_fused_metadata_layout()
        test_routed_fused_small_expert_set()
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
