"""Prefetch raw DY and local Y while retaining the original dot-product order."""
from pathlib import Path
import hashlib,json
rel='deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh'
s=(Path('work/next138/ilp4')/rel).read_text()
start=s.index('    } else if constexpr (Op == PeerOperation::ExpertLocalGradients ||')
end=s.index('    } else if constexpr (Op == PeerOperation::CountRoutes)',start)
part=s[start:end]
begin=part.index('            for (int64_t column = lane; column < width; column += 32) {')
finish=part.index('            #pragma unroll\n            for (int delta = 16;',begin)
old=part[begin:finish]
body=old[old.index('                const auto* d ='):old.rfind('            }')]
for ilp in (2,4,8):
 new='''            constexpr int gradient_ilp = ILP_VALUE;
            #pragma clang loop unroll(disable)
            for (int64_t base = lane; base < width; base += 32 * gradient_ilp) {
                uint4 dy_values[gradient_ilp], y_values[gradient_ilp];
                #pragma unroll
                for (int j = 0; j < gradient_ilp; ++j) {
                    const int64_t column = base + j * 32;
                    dy_values[j] = valid && column < width ? dy[column] : uint4{};
                    y_values[j] = valid && column < width ? peer_pointer<uint4>(p, 4)[row * width + column] : uint4{};
                }
                #pragma unroll
                for (int copy_j = 0; copy_j < gradient_ilp; ++copy_j) {
                    const int64_t column = base + copy_j * 32;
                    if (column < width) {
                        const uint4 raw_dy = dy_values[copy_j];
                        const uint4 raw_y = y_values[copy_j];
BODY
                    }
                }
            }
'''.replace('ILP_VALUE',str(ilp)).replace('BODY','\n'.join('        '+line for line in body.rstrip().splitlines()))
 result=s[:start]+part[:begin]+new+part[finish:]+s[end:]
 p=Path(f'work/next139/ilp{ilp}')/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(result)
 print(json.dumps(dict(ilp=ilp,sha256=hashlib.sha256(result.encode()).hexdigest())))
