"""Generate isolated header-only ILP candidates; preserve the installed baseline."""
from pathlib import Path
import hashlib, json

path = 'deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh'
source = (Path('work/implementation')/path).read_text()
begin = source.index('    } else if constexpr (Op == PeerOperation::StageRawWeightedDy) {')
end = source.index('    } else if constexpr (Op == PeerOperation::ExpertLocalGradients ||', begin)
section = source[begin:end]
old = '''            for (int64_t column = lane; column < width; column += 32) {
                const uint4 raw = valid ? source[column] : uint4{};
                const auto* input = reinterpret_cast<const __ppu_bfloat16*>(&raw);
                uint4 scaled{};
                auto* output = reinterpret_cast<__ppu_bfloat16*>(&scaled);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    output[j] = __float2bfloat16(__bfloat162float(input[j]) * score);
                peer_pointer<uint4>(p, 3)[row * width + column] = raw;
                peer_pointer<uint4>(p, 4)[row * width + column] = scaled;
            }'''
assert section.count(old) == 1
for ilp in (2,4,8):
    new = '''            // Issue independent peer reads before consuming them. Keep the
            // BF16 multiplication and the two destination layouts unchanged.
            constexpr int copy_ilp = ILP_VALUE;
            #pragma clang loop unroll(disable)
            for (int64_t base = lane; base < width; base += 32 * copy_ilp) {
                uint4 values[copy_ilp];
                #pragma unroll
                for (int j = 0; j < copy_ilp; ++j) {
                    const int64_t column = base + j * 32;
                    values[j] = valid && column < width ? source[column] : uint4{};
                }
                #pragma unroll
                for (int j = 0; j < copy_ilp; ++j) {
                    const int64_t column = base + j * 32;
                    if (column < width) {
                        const uint4 raw = values[j];
                        const auto* input = reinterpret_cast<const __ppu_bfloat16*>(&raw);
                        uint4 scaled{};
                        auto* output = reinterpret_cast<__ppu_bfloat16*>(&scaled);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k)
                            output[k] = __float2bfloat16(__bfloat162float(input[k]) * score);
                        peer_pointer<uint4>(p, 3)[row * width + column] = raw;
                        peer_pointer<uint4>(p, 4)[row * width + column] = scaled;
                    }
                }
            }'''.replace('ILP_VALUE', str(ilp))
    result = source[:begin]+section.replace(old,new)+source[end:]
    target = Path(f'work/next138/ilp{ilp}')/path
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(result)
    print(json.dumps(dict(ilp=ilp,path=str(target),sha256=hashlib.sha256(result.encode()).hexdigest())))
