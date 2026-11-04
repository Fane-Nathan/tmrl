# Reference verification for the sensor-focused pilot draft

The prior verification ledger is copied as `PRIOR_REFERENCE_VERIFICATION.md` for the twelve retained sources. Their role is background and method context; the pilot does not claim to reproduce each cited algorithm. Deployment-only TMRL and DrQ-v2 entries were removed from this sensor-focused bibliography.

Additional primary sources verified through BrowserOS on 16 September 2026:

| Key | Verified metadata | Supported role |
| --- | --- | --- |
| primacy | Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville. The Primacy Bias in Deep Reinforcement Learning. ICML 2022, PMLR 162:16828-16847. [Primary proceedings](https://proceedings.mlr.press/v162/nikishin22a.html). | Early-experience bias can affect later learning; this does not diagnose the author's earlier plateau. |
| plasticity | Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton. Loss of plasticity in deep continual learning. Nature 632:768-774 (2024). DOI 10.1038/s41586-024-07711-7. [Publisher article](https://www.nature.com/articles/s41586-024-07711-7). | Ability to continue learning is distinct from simply retaining earlier performance; the pilot does not implement or validate continual backpropagation. |
| clear | David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, Gregory Wayne. Experience Replay for Continual Learning. NeurIPS 32 (2019). [Primary proceedings](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html). | Replay is relevant to continual RL; CLEAR also uses objectives not implemented by the pilot. |

Verification covered metadata and relevant abstracts; selected sections of the Nature article were also consulted. This is not a claim of a full review of every cited paper or human-author verification.

All 15 bibliography entries are cited. Eleven fall in the conservative 2022-2026 calendar window: **11/15 = 73.3%**. The older foundational sources are SAC (2018), REDQ (2021), RL2 (2016), and CLEAR (2019). Recompute this fraction if references are changed.

Observation-wrapper semantics were additionally checked against [Gymnasium's official documentation](https://gymnasium.farama.org/api/wrappers/observation_wrappers/). Sensor-delay reset padding and the distinction between observation and action transformations are explicitly documented in the protocol and verified by behavioral tests.

The template files are unmodified LNCS 2.20 and `splncs04.bst` from the conference-linked archive already audited in the copied `template_manifest.json`.
