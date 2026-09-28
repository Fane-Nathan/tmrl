# Revised ICoDiT manuscript source

This is an **anonymous review draft**, not a submitted paper. Read `../REVISION_STATUS.md` for the remaining author and conference-policy decisions. Preserve the original manuscript and experiment records.

## Contents

- `manuscript.tex`, `references.bib`: revised manuscript and verified bibliography.
- `llncs.cls`, `splncs04.bst`: unmodified files from the conference-linked `full-paper-template.zip`, downloaded through BrowserOS on 15 September 2026 and inspected on 16 September. Its nested LaTeX archive supplies LNCS version 2.20, dated 10 March 2018.
- `history_effects.pdf`: vector plot included in the paper; PNG is a preview.
- `return_rows.tex`, `effect_rows.tex`: generated table macros.
- `data/episodes.csv`: all 1,800 archived evaluation records, including seed, condition, mode, episode seed, return, length, and checkpoint hash.
- `statistics.json`, `data/seed_means.csv`: regenerated results.
- `data/checkpoint_audit.json`: verification of original checkpoint hashes and validation selections.
- `data/context_position_audit.json`: comparison with seeded initialization of the current local implementation.
- `code_snapshot/`: current benchmark code and protocol, frozen during manuscript preparation. These files were not tracked in the historical run commit; this snapshot does not repair that gap.

## Build the manuscript

Use a recent TeX Live installation with `amsmath`, `amssymb`, `graphicx`, `booktabs`, `xurl`, and `hyperref`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The included class and bibliography style take precedence over a different installed LNCS version. The manuscript uses the conference template's default Computer Modern fonts and page geometry. BibTeX sorts the bibliography in the publisher style.

## Recompute the tables and figure

Install the versions in `requirements-analysis.txt`, then:

```powershell
python reproduce_analysis.py
```

This uses only `data/episodes.csv`, resamples the eight training-seed differences, and regenerates the statistics, table macros, and figure. It does not query W&B, run the simulator, or perform training. A source-verification option accepts the original `fault_benchmark_results` directory and checks actual checkpoint file hashes and validation records:

```powershell
python reproduce_analysis.py --source PATH_TO_ORIGINAL_RESULTS
```

The episode returns were rounded to four decimals when originally written. Regenerated means and intervals can differ from higher-precision JSON summaries in the last decimal places; reported one-decimal values match.

## Reproduce the positional-embedding diagnostic

From the original research workspace, using its installed Torch/Gymnasium environment:

```powershell
python output/icodit_2026/revised/audit_context_positions.py --workspace .
```

Original trained checkpoints are intentionally not duplicated into this lightweight source package. The script reads the existing checkpoints. It compares them with seeded initialization of the current implementation and makes no claim that the historical scripts were captured exactly.

## Prism / other LaTeX editors

The LaTeX import ZIP contains the manuscript, bibliography, figure, generated table macros, class, and bibliography style at the archive root. Set `manuscript.tex` as the main document. It is a separate revision; it does not replace the original `complete paper.tex` project. The imported [ICoDiT 2026 Review Draft](https://prism.openai.com/?u=7722d861-deb7-4604-ac24-51ac72ba8be0&pg=1&m=manuscript.tex) compiled successfully on 16 September 2026; its source hash matches this manuscript and its PDF has 13 pages.

## Scientific scope

The primary history effect is negative with an interval spanning zero. Neither the former three-seed positive headline nor the unverified TrackMania A/C comparisons is retained. The saved 32-token training / 64-token inference mismatch remains a limitation of the experiment, not something that prose editing can correct. A future evaluation must retain a separate result set.
