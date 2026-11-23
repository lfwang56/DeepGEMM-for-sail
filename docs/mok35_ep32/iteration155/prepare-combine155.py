"""Vectorize aligned BF16 top-k reduction; keep the scalar view/tail contract."""
from pathlib import Path
import difflib,hashlib,json
b=Path('work/implementation');d=Path('work/next155');patch=[]
def save(rel,old,new):
 p=d/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(new)
 patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
rel='deep_gemm/include/deep_gemm/impls/moe_training.cuh';s=(b/rel).read_text()
needle='''        // dims: chunk tokens, H, top-k, input tokens, chunk start, apply scores.
        const int64_t token = index / p.dim[1], column = index % p.dim[1];'''
replacement='''        // dims: chunk tokens, H, top-k, input tokens, chunk start, apply scores.
        const bool vectorized = p.dim[1] % 8 == 0 &&
            (reinterpret_cast<uint64_t>(p.ptr[0]) & 15u) == 0 &&
            (reinterpret_cast<uint64_t>(p.ptr[2]) & 15u) == 0;
        if (vectorized) {
            const int64_t vectors = p.dim[1] / 8;
            const int64_t token = index / vectors, column = (index % vectors) * 8;
            const int64_t source_token = token + p.dim[4];
            if (token >= p.dim[0] || source_token >= p.dim[3]) return;
            float values[8] = {};
            // Keep slot accumulation order, sharing each score/address across
            // eight features and issuing aligned 16-byte memory operations.
            for (int slot = 0; slot < p.dim[2]; ++slot) {
                const float score = p.dim[5] ? pointer<float>(p, 1)[source_token * p.dim[2] + slot] : 1.f;
                const uint4 raw = *reinterpret_cast<const uint4*>(pointer<Bf16>(p, 0) +
                    (token * p.dim[2] + slot) * p.dim[1] + column);
                const auto* input = reinterpret_cast<const Bf16*>(&raw);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    values[j] += __bfloat162float(input[j]) * score;
            }
            uint4 packed;
            auto* output = reinterpret_cast<Bf16*>(&packed);
            #pragma unroll
            for (int j = 0; j < 8; ++j) output[j] = __float2bfloat16(values[j]);
            *reinterpret_cast<uint4*>(pointer<Bf16>(p, 2) + source_token * p.dim[1] + column) = packed;
            return;
        }
        const int64_t token = index / p.dim[1], column = index % p.dim[1];'''
assert s.count(needle)==1;save(rel,s,s.replace(needle,replacement))
rel='csrc/apis/moe_training.hpp';s=(b/rel).read_text();a=s.index('        case Operation::Combine:');z=s.index('        case Operation::Activation:',a);part=s[a:z]
old='            work = dims[0] * dims[1];'
new='''            work = dims[0] * dims[1];
            if (dims[1] % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[2].data_ptr()) % 16 == 0)
                work /= 8;'''
assert part.count(old)==1;save(rel,s,s[:a]+part.replace(old,new)+s[z:])
(d/'combine155.patch').write_text(''.join(patch))
print(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file()},indent=2))
