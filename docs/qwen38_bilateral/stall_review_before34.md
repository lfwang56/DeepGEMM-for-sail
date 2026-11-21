# Stall review after iterations31–33

2026-11-21UTC. Three successive directions failed the>=3% complete-step gate:
WG PU/AIU issue scheduling31, raw-DY replay overlap32, low-register inline
input assistants33. Defaults still producer G8 ILP8, serial WG/DY,~97.4ms.
No physical performance floor has been established.

Re-read all controlled results and the new32Asys disjoint-interval analysis.
RawDY copy overlaps5.571ms but replayFC1 slows11.784->17.176ms. WG concurrency
creates19.425ms of compute/mixed task intersection but raises all long-kernel
spans.33actualEP32 inline=100.086ms and assist8/16/32/39=104.234/103.385/
105.024/104.217ms versus producer97.401ms. Strong DeepEPearlyDY120.065ms.
Interfered SN2 control attempts0/1 are excluded; clean controls and all33
candidates share SN1219220216217. No precision, tolerance or baseline weakening.

Frozen33 single-PPU local-pointer Asys uses the actual H8192/I2048/E16,K10,
T4096 source padded to8192 and ring65536 (167active M tiles). Aux8 wins32
tiles. Main inline GEMM is240regs/thread,512threads,2176static+131072dynamic
shared bytes. Aux uses64regs,128threads,2176static bytes. Across3FC1 replays,
GEMM spans46.881ms,copy20.286ms,intersection20.092ms. This is actual task
concurrency but local pointers, NOT ICN performance. Benchmark's last-window
assisted count0 is uninformative because empty routed windows reset it.

Source review: auxiliary operation16 divides64-bit vector positions by runtime
width for every vector, while producer has compile-time K. Specializing H/BM
can remove repeated division without altering BF16 copies/publication. Also the
inline fallback keeps the copy path in every compute CTA's240-register binary.
A separately compiled consumer without inline fallback could shed those
registers, with launch occupancy limiting compute to(numCUs-1)*occupancy and
leaving room for a low-resource copy CTA to guarantee progress. Validate native
resources and all per-tile readiness/dependency/timeout rules; do not rely on
NVIDIA scheduling assumptions. Both changes require controlled ablations.

Fresh primary web research:
- https://help.aliyun.com/zh/document_detail/3011255.html (SDK2.0)
- https://help.aliyun.com/zh/document_detail/3030339.html (SDK2.1)
- https://help.aliyun.com/en/document_detail/2971904.html (Asys tracing)
- https://help.aliyun.com/en/document_detail/2871816.html (Acu replay)
Official releases say HGGC_EXCLUSIVE_STREAMS maps stream groups to hardware
queues. Installed2.2 libhggc.so contains EXCLUSIVE_STREAMS/exclusiveStreams,
but neither current headers nor searches established its syntax/group APIs.
Do not invent values or attribute a lack of overlap to queue mapping.
Asys documentation uses event-duration sums for stream percentages; use measured
wall unions/intersections. Range Acu replay can preserve parallel kernels;
ordinary per-kernel replay is a resource diagnostic, not overlap evidence.

Read PPU wiki hggc-kernel-development.md and tools/asight-systems.md: explicit
streams/events, stream-ordered lifetime rules, graph-internal stream assignment,
profiling perturbation and installed-version boundary. Reviewed existing31
native ACTLIZE issue-order and group-contention findings; no new magic hardware
instruction or unsupported occupancy rule is inferred.

Decision: investigate specialization and separate resource footprints before
another candidate; keep source frozen while final baseline validation runs.
