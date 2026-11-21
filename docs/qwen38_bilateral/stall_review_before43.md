# Reassessment after40–42

Re-read the complete accepted four-host matrices and numerical/occupancy evidence.
40producer85.411 vsbestexternal88.758; pacing/gridchanges didnotbeatproducer.
41WGstage3 gives83.195 vs84.226 (1.22%,below3%major-win criterion), withthe same
WGgeometryonElastic106.848.42currentcontrol83.174 vsstrongestElasticearly105.704.
CoupledWG16/DGrad23parallel89.682 vsidentical-budgetserial116.296 is a real
22.9% overlap improvement, yetstill6.509ms slowerthanbestproducer. TinyWGradtiles
with32–74KBsharedmemory,128threads andpositiveoccupancy3–8allpassed fullFP32,
mutablelengthgraphs andactualEP32checks; their serial109–122ms andparallel184–335ms
are regressions. Per-kerneloccupancy doesnotguarantee jointresidency, aCUaffinity
partition, or usefulinstructionoverlap. No physicalfloor established.

Refreshed primary T-Head searches2026-11-21 on shared-memory groups, GEMM and
communication/compute fusion (https://developer.t-head.cn/blog/index.html).
The official FlashMLA/group-contention and GEMM/wave articles are workload-specific;
installed SDK2.2.0/hardware validation remains authoritative, not NVIDIA assumptions.
The dynamic blog36 URL was unavailable to the web reader; do not claim a new
hardware reproduction from that failed open. Existing Asight37 shows local44.16ms,
includingWG16.92 andzero cross-kernelintersection. Frozen42profiles are being
captured for the best serial and paired-budget concurrent paths.

Choose a different direction43: fuse raw peer DY reads, mandatory BF16 weighting
and per-tile FP32 score-dot partials into replay FC2's epilogue. A small follow-up
reduces partials and returns dScore, avoiding the full expert-Y write/read and
standalone H-wide DY/score kernel. Preserve BF16 Y/weighted-DY boundaries, complete
remat, allDX/DW and dropless capacities. Use a separate optional GPU argument and
new pointwise opcode; do not alter PeerGemmParams or old kernel ABI. FP32 score
summation is reassociated; require predeclared graph RMS<1e-5 and elementwise
rtol2e-5/atol2e-6 for dScore, unchanged exactY/DX and DW thresholds, then every-element
independentFP32 and actual32-rank/fullC4096skew checks before timing. Failure is
not permission to weaken thresholds. Compare the strongest installed DeepEP.
