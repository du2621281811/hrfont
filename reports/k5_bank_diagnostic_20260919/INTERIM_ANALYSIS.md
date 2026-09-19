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
