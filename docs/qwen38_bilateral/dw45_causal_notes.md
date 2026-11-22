# DW resource and admission diagnosis, 2026-11-22

Chip ZW-M890P, ppu0015, SDK2.2.0-663c22, Torch2.11.0+v0.2.0.ppu2.2.0,
Asight2.2.0,39visibleCUs, BF16. Actual same-domain EP32 four-host runs.
Qwen routed layer H8192/I2048/E512/top10,T4096,C4096,G8,BM256,MGroup/WGrad3.

The old 16.6 versus37.8ms comparison changed both concurrency and CTA
budgets: full39versusWG16/dGrad23. CTA budgets do not provide CU affinity.
Matched immutable-wheel fullserial/cappedserial/cappedparallel/fullparallel
profiles gave DW16.184/38.071/38.218/31.772ms. Reducing the CTA budget is
the largest cause of that original discrepancy; same-capped concurrency
adds only0.4% to the summed DW kernel intervals.

An optional independent launch argument records native __ppu_read_clock64
at entry and exit of every CTA, with a final CTA barrier. No subtraction
of clock values from different CUs. This measures elapsed time after CTA
entry, including subsequent stalls/preemption; it is not pure MMA activity.
The probe is bitwise equal to uninstrumented WGrad on identical packed
inputs, including both complete Qwen DW output shapes. Complete routed
plans with independently rebuilt atomic permutations use the established
1e-5 DW RMS gate and observed approximately3.5e-7; Y/DX/dScore are bitwise.

Three-graph probe profiles, last graph42calls matched to Asys:

| Schedule | DW kernel span sum ms | Sum max same-CTA elapsed cycles |
| --- | ---: | ---: |
| full serial |15.928756|20,609,680|
| full parallel |25.362384|21,106,442|
| capped serial |37.871661|49,134,698|
| capped parallel |37.400049|48,504,886|

The full parallel CTA elapsed sum rises2.4%, far less than the59.2%
increase in summed Asys kernel intervals. This supports dispatch/admission
waiting or delayed CTA waves as a major contributor to the interval growth,
rather than a claim that resident GEMM arithmetic slows by the same factor.
Exact clock-domain/entry alignment and hardware activity counters remain
necessary for a finer attribution. These are separately profiled executions;
do not substitute kernel interval sums for unprofiled complete-step latency.
A short WGrad output-tile pipeline is motivated by these results.

Evidence: outputs/qwen38-round2/dw-cap-profile-summary.json,
iter45-cta-probe-provenance.json,iter45-cta-vs-asys.json in the campaign
workspace. Immutable previous delivery remains fa124e6359cc.
