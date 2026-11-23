"""Explicit opt-in JIT parameters for gradient prefetch and a controlled MMA A/B."""
from pathlib import Path
import difflib,json,hashlib

base=Path('work/implementation');dest=Path('work/next143');patch=[]
def save(rel,old,new):
 p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(new)
 patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
rel='deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh'
s=(base/rel).read_text();n=(Path('work/next139/ilp2')/rel).read_text()
assert 'uint32_t InputSlices = 1>' in n
n=n.replace('uint32_t InputSlices = 1>','uint32_t InputSlices = 1, uint32_t GradientCopyIlp = 1>')
n=n.replace('constexpr int copy_ilp = 4;','constexpr int copy_ilp = GradientCopyIlp;').replace('constexpr int gradient_ilp = 2;','constexpr int gradient_ilp = GradientCopyIlp;')
n=n.replace('    if (block_index < 0) block_index = blockIdx.x;','    static_assert(GradientCopyIlp == 1 || GradientCopyIlp == 2 || GradientCopyIlp == 4 || GradientCopyIlp == 8);\n    if (block_index < 0) block_index = blockIdx.x;',1)
save(rel,s,n)
rel='csrc/apis/peer_moe_training.hpp';s=(base/rel).read_text()
needle='        const bool cache_fp8_scales ='
at=s.index(needle)
n=s[:at]+'''        const bool gradient_stage = args.operation == int(PeerOperation::StageRawWeightedDy);
        const bool gradient_pack = args.operation == int(PeerOperation::ExpertLocalGradients) ||
                                   args.operation == int(PeerOperation::ExpertLocalGradientsQuantized);
        const int gradient_ilp = get_env<int>(gradient_stage ? "DG_MOE_DY_COPY_ILP" : "DG_MOE_GRADIENT_COPY_ILP", 1);
        DG_HOST_ASSERT(gradient_ilp == 1 || gradient_ilp == 2 || gradient_ilp == 4 || gradient_ilp == 8);
'''+s[at:]
needle='       input_tiles ? fmt::format('
assert n.count(needle)==1
n=n.replace(needle,'       gradient_stage || gradient_pack ? fmt::format(", false, 0, 1, {}", gradient_ilp) :\n'+needle)
save(rel,s,n)
rel='csrc/jit/compiler.hpp';s=(base/rel).read_text()
old='''        return get_ppu_arch() == AC_PPU0015 && name == "bf16_gemm_rs" ? "-ppu-pref-mma-reuse=true" : "";'''
assert s.count(old)==1
new='''        const int peer_reuse = get_env<int>("DG_MOE_PEER_MMA_REUSE", 0);
        DG_HOST_ASSERT(peer_reuse == 0 || peer_reuse == 1);
        const bool peer = name.rfind("bf16_peer_", 0) == 0;
        return get_ppu_arch() == AC_PPU0015 &&
               (name == "bf16_gemm_rs" || (peer && peer_reuse)) ? "-ppu-pref-mma-reuse=true" : "";'''
save(rel,s,s.replace(old,new))
(dest/'candidate.patch').write_text(''.join(patch))
print(json.dumps({'files':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}},indent=2))
