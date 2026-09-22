# Experiments 节完整 LaTeX 草稿（2026-09-23）

口径：训练配方 K6-B；数据写 V3（补充集 train-only、不进 bank，泄露只说一句设定）；对照组四层；像素指标按数字照报、正文不解释；度量有效性研究以「指标」为主语（属评测贡献）；人评三处落点。数字占位一律 `--`（未填时排版显示为短横线，fill 时整体替换）。main.tex 未改。

```latex
\section{Experiments}
\label{sec:experiments}

\subsection{Setup and protocol}
The corpus combines the main 260-typeface corpus with a 315-typeface supplement used for training only.
The split keeps 228 training, 16 validation, and 16 test families in the main corpus, and every family provides 295 target characters and 338 Chinese reference characters.
The supplement is used for training only and is never added to the retrieval bank, which stays frozen on the donor list of the main corpus.
Protocol A renders RGB glyphs at $96\times96$ with a font size shared by all characters of a family and a neutral Noto Sans CJK Regular content image; the clean mapping retains 56{,}429 pairs from the main corpus and 41{,}663 from the supplement.
Evaluation uses a stratified panel of 47 characters covering Latin, diacritics, kana, and phonetic symbols, which retains 704 test pairs after validity filtering.
Reference budgets are nested sets of 1, 2, 4, and 8 characters from one ordered manifest, and all comparisons share the reference set, the bank budget, and the condition interface.

Identity is measured by character recognition accuracy and cross-script consistency by the human protocol and the evaluator below; L1, SSIM, and LPIPS are paired reconstruction diagnostics.
The human protocol shows the Chinese references and one generated glyph at $96$\,px without the target design, the method name, or any score, and collects a five-point style-compatibility rating together with a secondary plausibility rating; each image receives three valid ratings.
Comparisons are paired per item over shared references, sampler, and noise, and report confidence intervals with sign and Wilcoxon tests corrected by Holm.

\paragraph{Training.}
The model starts from an RSI-free checkpoint trained for 10k updates and runs for 20{,}000 updates on eight GPUs with a global batch of 64.
The schedule warms up over 500 updates, holds a plateau until 5{,}000, and decays cosine to 10\% of the peak rate; the denoiser and the structural modules use a peak rate of $2\times10^{-5}$ and the reader and the routing read-outs $10^{-4}$, with AdamW, $\beta=(.9,.999)$, weight decay $.01$, and FP16 arithmetic.
Model weights are averaged with an EMA factor of $\min(.999,1-1/(u+1))$.
The objective keeps the backbone perceptual term at $0.01$, an appearance term at $0.01$ ramped by $\beta=.8\min(u/1000,1)$, and a detail term at $0.05$; condition dropout uses a joint rate of $0.02$ and a source drop of $0.05$.
Resolved settings appear in Appendix~\ref{app:repro}.

\subsection{Cross-script consistency evaluator}
The evaluator scores how closely a generated glyph fits the family observed in the Chinese references.
It builds a shared representation across scripts, aggregates the reference context, and calibrates a membership head by temperature scaling, so the score is comparable across families.
It is trained on rendered glyphs of the corpus, independently of the generation model.
We report its consistency across scripts, family discrimination, agreement with known targets, and behaviour under mismatched references, together with per-image agreement with human ratings (Appendix~\ref{app:evaluator}).

\subsection{Main comparison}
\begin{table}[t]
\centering\small
\caption{Main comparison under the cross-script protocol. Identity is character recognition accuracy, style preference follows the human protocol, and family compatibility is the evaluator score; higher is better for all three.}
\label{tab:main}
\begin{tabular}{lrrrr}
\toprule
Method & Identity $\uparrow$ & Style pref.\ $\uparrow$ & Family $\uparrow$ & LPIPS $\downarrow$\\
\midrule
FontDiffuser & -- & -- & -- & --\\
FTransGAN & -- & -- & -- & --\\
CF-Font & -- & -- & -- & --\\
FSFont & -- & -- & -- & --\\
CRP (fixed weights) & -- & -- & -- & --\\
\method & -- & -- & -- & --\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Control studies}
\begin{table}[t]
\centering\small
\caption{Control studies on the full model. The first block removes one condition; the second replaces one design choice with a simpler form.}
\label{tab:controls}
\begin{tabular}{llrr}
\toprule
Study & Control & Identity $\uparrow$ & Family $\uparrow$\\
\midrule
\multicolumn{4}{l}{\emph{Removal}}\\
Structural condition & without the structural offset & -- & --\\
Appearance condition & without the appearance condition & -- & --\\
\midrule
\multicolumn{4}{l}{\emph{Substitution}}\\
Combination form & routed vs fixed prior weights & -- & --\\
Structure form & neutral-referenced offset vs direct donor rendering & -- & --\\
\bottomrule
\end{tabular}
\end{table}

\subsection{What the metrics measure}
\label{sec:metric-study}
L1, SSIM, and LPIPS measure agreement with one realization of the target character; the compatibility score and the recognition accuracy are conditioned on the reference set.
We compute all measures on the same panel and compare their orderings per sample.
L1 and the compatibility score order the compared methods at Spearman $\rho = $ --, so the two quantities carry different information about a generated glyph.
Against the human ratings, L1 agrees at $\rho = $ -- and the compatibility score at $\rho = $ -- ($n = $ --).
Two controls show what each family of measures responds to: weight siblings of one design differ by L1 $= $ -- while both belong to the family, and glyphs of a different family at comparable pixel distance score below the references (Figure~\ref{fig:metric-roles}).

\begin{table}[t]
\centering\small
\caption{Measures on the same panel. The first three rows are point measures against a single realization; the remaining rows are conditioned on the reference set.}
\label{tab:metric-roles}
\begin{tabular}{llrr}
\toprule
Measure & Conditioned on & Agreement with pixel measures & Agreement with human ratings $\uparrow$\\
\midrule
L1 & one realization & -- & --\\
SSIM & one realization & -- & --\\
LPIPS & one realization & -- & --\\
Style feature distance & reference set & -- & --\\
Stroke and skeleton similarity & reference set & -- & --\\
Compatibility score & reference set & -- & --\\
Recognition accuracy & target character & -- & --\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
% \input{figures/metric_roles.tex}  % 待生成
\caption{Measures against family membership. Weight siblings of one design and glyphs of a different family at comparable pixel distance.}
\label{fig:metric-roles}
\end{figure}

\subsection{Human evaluation}
Human ratings are the primary evidence for cross-script consistency, and the evaluator is validated against them in Section~\ref{sec:metric-study}.
Ratings cover the comparisons of the main table on a sampled subset of the panel; the ablation arms are evaluated with the automatic axes.

\begin{table}[t]
\centering\small
\caption{Human evaluation. Style preference is the primary dimension and plausibility the secondary one. Win rates are pairwise against \method{} with the reference set held fixed, and $p$-values are corrected by Holm.}
\label{tab:human}
\begin{tabular}{lrrrr}
\toprule
Comparison & Style pref.\ mean $\uparrow$ & Win rate (\%) & $n$ & $p$\\
\midrule
FontDiffuser & -- & -- & -- & --\\
FTransGAN & -- & -- & -- & --\\
CF-Font & -- & -- & -- & --\\
FSFont & -- & -- & -- & --\\
CRP (fixed weights) & -- & -- & -- & --\\
\method & -- & -- & -- & --\\
\bottomrule
\end{tabular}
\end{table}

Inter-rater agreement (Krippendorff's $\alpha$) is -- for style preference and -- for plausibility, and the protocol, the sampling rule, and the per-dimension distributions appear in Appendix~\ref{app:human}.

\subsection{Reference budgets and qualitative analysis}
Reference-budget curves evaluate one checkpoint at 1, 2, 4, and 8 references for \method{} and for the external methods at the same budget, and show how identity, appearance, and family coherence change as more observations arrive.
Qualitative comparisons cover ordinary stroke styles, decorative treatments, and difficult cross-script realizations.

\begin{figure}[t]
\centering
% \input{figures/budget_curves.tex}  % 待生成
\caption{Reference-budget curves. Identity, appearance, and family coherence as a function of the number of references, for the full model and the external methods at the same budget.}
\label{fig:budgets}
\end{figure}

\begin{figure}[t]
\centering
\input{figures/qualitative_layout.tex}
\caption{Qualitative comparison. Rows cover stroke style, decorative treatment, and difficult cross-script realizations.}
\label{fig:qualitative}
\end{figure}
```

## 占位清单（待填）

| 占位 | 内容 | 依赖 |
|---|---|---|
| `tab:main` 6×4 | 外部方法 + CRP (fixed weights) + ours | 外部复现数字来源、我们固定协议推理 |
| `tab:controls` 4×2 | removal ×2、substitution ×2 | 对照臂训练/推理结果 |
| `tab:metric-roles` 两列 | 与像素度量的一致性、与人评的一致性 | 前者零人评即可算；后者待人评 |
| `fig:metric-roles` | 正/负对照图 | 变体/外部字体池，零训练 |
| `tab:human` 四列 | 均分、胜率、n、p | 人评结果 |
| 四个 $\rho$ 与 $n$ | 去相关 + 人评一致性 | 人评结果 |
| `fig:budgets` / `fig:qualitative` | 预算曲线 / 定性面板 | 固定协议推理产物 |
