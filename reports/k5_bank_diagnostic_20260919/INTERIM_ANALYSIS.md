# K5 bank diagnostic: intermediate findings

2026-09-20. This is not final representation-audit acceptance.

576 images and hashes previously passed. Deduplicating repeated ordinary controls leaves36 unique font/character/seed cases per model. Selection traces confirm actual10,50,417 donors for Top10/Top50/all respectively; the all condition was not silently limited to10.

| Change from Top10 | A mean pixel difference | B mean pixel difference |
|---|---:|---:|
| all417, original alpha |0.00010248|0.00012559|
| all417, uniform |0.00010549|0.00012760|
| Top10 uniform |0.00001808|0.00002411|
| Top50 |0.00006028|0.00005928|
| Delta-off |2.39972|1.68036|

Units are uint8 intensity levels (0..255), not normalized L1 and not training loss. These are paired image differences, not quality improvements. Donor/alpha changes have extremely small effects in this fixed panel; switching the path off has much larger effects. Thus this panel does not support an immediate Top10→all-bank remedy. It does not prove donor retrieval correct, nor that Delta encodes transferable geometry.

Additional K6-0 router traces on frozen K5-B show nonzero spatial routing variation and differences from uniform alpha. K6_0_ROUTER_SUMMARY aggregates these descriptive traces. Its neutral/adversarial names are operational proxies, not validated Western style categories. Branch offset norms are pre-gate: do not compare the raw Es norm to the final combined Ec+gatedEs norm as if they were the same quantity.

A bounded direct probe (FZBangSKLTJW A,4shot,seed3407,first denoising t999) keeps real hidden activations fixed and zeroes or horizontally flips donor spatial features. All inspected A/B branches show nonzero offset changes. Therefore the donor interface is not identically disconnected in that probe. This remains one input/time probe, not final-image evidence of style recovery or a complete temporal representation audit. K5-B gate values are recorded separately in K5-B_OFFSET_PROBE.json.

Remaining: match perturbation effects through mixed delta, combined offset and final output across representative denoising times; audit representation coverage/redundancy and GT-consistent style effects. No training recipe or retrieval policy changed. K6 formal training remains held pending explicit objective-path review.

## Additional audit

Original alpha already nearly uniform in this panel: Top10 entropy effective count9.99989/10 and mean maximum weight0.100898; all-bank416.983/417. Consequently uniform-alpha is a very weak intervention here, not evidence that arbitrary priors would have no effect. Repeated ordinary-font outputs with identical refs/seeds differ by up to one uint8 level (mean across repeats0.000007535); tiny nonzero image differences should not be overinterpreted. Selection and repeat statistics are in SELECTION_AUDIT.json. Frozen-hidden offset probes were extended to five calls along the same real glyph trajectory (OFFSET_TEMPORAL_PROBE_DONE.json), preserving the single-glyph scope; full representation audit remains outstanding.

## Centered bank feature geometry and dual-offset cancellation

Raw frozen spatial feature audit covers A/g/3 with the same legal417-donor bank and target/reference mask. Across48/24 scales, Ec effective rank22.6–50.8, Es65.8–91.9. The selected10 donors' unrestricted linear span captures31.0–46.4% of Ec centered variance and19.3–23.4% of Es. These numbers are not semantic style coverage: signed linear projection is more permissive than the router's convex mixture, and variance can contain nuisance information. They show that the raw bank is not trivially identical in these encoders, while the completed generation panel barely responds to changing its candidate subset.

A separate K5-B real-trajectory probe (FZBangSKLTJW A,seed3407,4shot) records both Ec offset and tanh(gate)*Es offset BEFORE addition and the actual combined output. Their cosine is approximately−0.9912 to−0.9999994 over6layers×5calls. The combined RMS divided by the sum of branch RMS is0.00061–0.07995. In the second inspected decoder block it is0.00061–0.00418. Thus strong branch cancellation is directly observed in this trajectory. Small gates alone would have concealed this because the unweighted Es offsets are large.

This is a candidate explanation for weak use of the dual branch, not a proof of global causality, nor an explanation of K5-A's similar candidate insensitivity. Offset regularization penalizes the combined offset, so a compensating solution is possible; assigning responsibility to the regularizer requires a controlled follow-up. No gate sign/normalization/loss was changed. An18-trajectory expansion (three fonts×A/g/3×two seeds,5calls) is now running in control/offset_combination_expanded_probe.py; results must be checked before generalizing.
