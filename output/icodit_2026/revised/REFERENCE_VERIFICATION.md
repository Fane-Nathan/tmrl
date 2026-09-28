# Reference verification ledger

Verified using BrowserOS on 15 September 2026. Verification covers the primary page's bibliographic metadata and abstract or repository description. It does not claim a full reading of every paper, a citation-index audit, or manual verification by the human authors. The manuscript makes only the broad literature claims supported by those readings.

| Key | Publication year | Primary source checked | Role in manuscript |
| --- | --- | --- | --- |
| rlpd | 2023 | [PMLR, Ball et al.](https://proceedings.mlr.press/v202/ball23a.html) | Prior offline data can assist online RL; controlled design choices matter. |
| crossq | 2024 | [ICLR 2024](https://iclr.cc/virtual/2024/poster/18699) | Normalization and update scheduling affect efficiency; not an implemented comparator. |
| redq | 2021 | [Author paper, publication note identifies ICLR 2021](https://arxiv.org/abs/2101.05982) | Ensemble critics and randomized target minimization. |
| rl2 | 2016 | [Duan et al., arXiv](https://arxiv.org/abs/1611.02779) | Observation/action/reward/termination conditioning; original across-episode formulation. |
| amago | 2024 | [Official ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7204434dcb9383a1454dc1e97e58ea9c-Abstract-Conference.html) | Off-policy transformer in-context RL. Correct authors: Jake Grigsby, Jim Fan, Yuke Zhu. |
| amago2 | 2024 | [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9f40d2612d0b3b6f6c2a77da21b1067f-Abstract-Conference.html) | Multi-task return-scale and optimization context; DOI from this page. |
| sac | 2018 | [PMLR, Haarnoja et al.](https://proceedings.mlr.press/v80/haarnoja18b.html) | Entropy-regularized off-policy actor–critic foundation. |
| droq | 2022 | [ICLR 2022](https://iclr.cc/virtual/2022/poster/6233) | Dropout/LayerNorm design context. Conference year, not 2021 preprint year. |
| ad | 2023 | [ICLR 2023](https://iclr.cc/virtual/2023/oral/12612), [author paper](https://arxiv.org/abs/2210.14215) | Distilling learning histories, contrasted with direct reward training. |
| ni | 2022 | [PMLR, Ni et al.](https://proceedings.mlr.press/v162/ni22a.html) | Recurrent baseline sensitivity to implementation and context. |
| empirical | 2024 | [JMLR 25(318), 1–63](https://jmlr.org/papers/v25/23-0183.html) | Empirical design, variability, baselines, and selection bias. |
| tmrl | Undated | [Official repository](https://github.com/trackmania-rl/tmrl) | Distributed real-time framework and TrackMania environment. Access date is not treated as publication year. |
| gym | 2024 | [Towers et al., arXiv v3](https://arxiv.org/abs/2407.17032v3) | Standard environment interface. Explicit 2024 version; newer v4 notes NeurIPS 2025 acceptance. |
| drqv2 | 2022 | [ICLR 2022](https://iclr.cc/virtual/2022/poster/6275) | Visual-control and augmentation context. Conference year, not 2021 preprint year. |

## Recency calculation

Using the conservative calendar window **2022–2026**, 10 references are recent and four are nonrecent (three older works plus the undated repository). Therefore **10 / 14 = 71.43%**. All 14 entries are cited in the text. The threshold can be broken by adding older citations: recompute after any bibliography change. Preserve foundational citations when scientifically necessary; do not change years or add irrelevant references to manipulate the ratio.

## Template sources

- [Conference template page](https://socs.binus.ac.id/icodit/post/templates-2/).
- [Springer author/editor information](https://link.springer.com/series/558/information-for-authors-and-editors), which explicitly includes CCIS in its scope.
- [Publisher LaTeX ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/27851904/data/LaTeX2e%20Proceedings%20Template%20ZIP), downloaded successfully through BrowserOS. Included class header: version 2.25, 3 September 2026.

The conference-linked archive was downloaded and inspected on 16 September 2026. Its LaTeX RAR contains LNCS version 2.20 (10 March 2018). The final revision uses that class, its default Computer Modern fonts, and its bibliography style unchanged. The newer publisher version was used only during preparation. See `data/template_manifest.json` for archive and file hashes.
