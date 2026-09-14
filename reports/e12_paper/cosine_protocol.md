# E12 Cosine Style Score protocol (teacher metric)

Primary evaluator metric (not membership):

\[
z_{zh}=\mathrm{mean}\{E(R_{zh})\},\quad
z_{en}=E(q_{en}),\quad
S_{\mathrm{cosine}}=\cos(z_{zh},z_{en}),\quad
S_{01}=(S_{\mathrm{cosine}}+1)/2
\]

- Encoder: `runs/e12_phi_s2_b_s3407/best.pt` (InfoNCE φ)
- Cache / usability: `artifacts/e12/cache_v0913_b` · `manifests/v0913_clean`
- Ref set \(R_{zh}\): 永和书风骨韵天地 (StyleImage)
- Query: Latin (+ digit for application tables)
- Font split: 70/15/15 by typeface group, seed 3407; retrieval/AUC on **test** families only
- Membership head: appendix / ablation only

Non-goals: aesthetics classifier; open-world font ID; dirty-data narrative.
