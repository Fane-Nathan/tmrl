"""Reproduce manuscript statistics/figure from the supplied episode records.

Run with Python 3.10+, numpy and matplotlib. No model training or network access.
The independent resampling unit is a training seed, not an evaluation episode.
"""
from pathlib import Path
import csv
import hashlib
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CONDITIONS = ['test_latency', 'test_dead', 'test_sign_flip']
LABELS = ['Action latency', 'Dead actuator', 'Sign flip']
SEEDS = list(range(42, 50))


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(12345)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(2000)])
    return [float(values.mean()), *np.percentile(means, [2.5, 97.5]).tolist()]


def main(source=None):
    data_dir = HERE / 'data'
    data_dir.mkdir(exist_ok=True)
    hash_checks = []
    records = []
    if source:
        for seed in SEEDS:
            for algorithm, count in [('transformer', 150), ('reactive', 75)]:
                run = source / f'{algorithm}_seed_{seed}'
                raw = run / 'final_test_raw_episodes.csv'
                with raw.open(newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                assert len(rows) == count, (run, len(rows))
                meta = json.loads((run / 'checkpoint_metadata.json').read_text())
                sha = hashlib.sha256((run / 'best_validation_checkpoint.pt').read_bytes()).hexdigest()
                assert sha == meta['checkpoint_sha256']
                assert all(r['checkpoint_hash'] == sha for r in rows)
                assert meta['selection_rule'] == 'argmax_validation_score'
                with (run / 'validation_metrics.csv').open(newline='') as f:
                    validation = list(csv.DictReader(f))
                selected = max(validation, key=lambda r: float(r['val_return']))
                assert int(selected['iteration']) == meta['best_iteration']
                assert abs(float(selected['val_return']) - meta['best_val_score']) < 1e-3
                hash_checks.append(dict(algorithm=algorithm, seed=seed, sha256=sha,
                                        selected_iteration=meta['best_iteration'],
                                        validation_rows=len(validation), verified=True))
                records.extend(rows)
        with (data_dir / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (data_dir / 'checkpoint_audit.json').write_text(json.dumps(hash_checks, indent=2))
    else:
        with (data_dir / 'episodes.csv').open(newline='', encoding='utf-8') as f:
            records = list(csv.DictReader(f))
    assert len(records) == 1800
    results = {}
    seed_rows = []
    for cond in CONDITIONS:
        means = {'adaptive': [], 'no_history': [], 'reactive': []}
        for seed in SEEDS:
            groups = {}
            for mode in means:
                rows = [r for r in records if r['fault_condition'] == cond
                        and int(r['seed']) == seed
                        and (r['mode'] == mode if mode != 'reactive'
                             else r['algorithm'] == 'reactive_robust_baseline')]
                # Some versions label the reactive mode differently; the algorithm
                # field identifies the recorded baseline unambiguously.
                assert len(rows) == 25, (seed, cond, mode, len(rows))
                ids = {(int(r['episode_id']), int(r['environment_seed'])) for r in rows}
                assert ids == {(i, 50000 + 100*i) for i in range(25)}
                assert all(int(r['episode_length']) == 1000 for r in rows)
                groups[mode] = float(np.mean([float(r['return']) for r in rows]))
                means[mode].append(groups[mode])
            seed_rows.append(dict(condition=cond, seed=seed, **groups,
                                  G=groups['adaptive']-groups['no_history'],
                                  D=groups['adaptive']-groups['reactive']))
        means['G'] = np.subtract(means['adaptive'], means['no_history']).tolist()
        means['D'] = np.subtract(means['adaptive'], means['reactive']).tolist()
        results[cond] = {k: {'values': v, 'mean_ci95': bootstrap(v)} for k, v in means.items()}
        results[cond]['positive_G_seeds'] = sum(x > 0 for x in means['G'])
    output = dict(independent_unit='training seed', n_seeds=8, episodes_per_cell=25,
                  total_records=1800, bootstrap_resamples=2000, bootstrap_seed=12345,
                  interval='pointwise 95% percentile bootstrap; no multiplicity correction',
                  data_precision='returns rounded to four decimals in archived CSV',
                  endpoints=results)
    (HERE / 'statistics.json').write_text(json.dumps(output, indent=2))
    with (data_dir / 'seed_means.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    for ax, metric, title in zip(axes, ['G', 'D'],
                                 ['(a) History effect G', '(b) Control advantage D']):
        for i, cond in enumerate(CONDITIONS):
            vals = results[cond][metric]['values']
            mean, lo, hi = results[cond][metric]['mean_ci95']
            ax.scatter(vals, i + np.linspace(-0.13, 0.13, 8), s=15,
                       color='0.58', zorder=3, label='Training seeds' if i == 0 else None)
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='D',
                        color='#164f78', markersize=4, capsize=4, linewidth=1.5,
                        zorder=4, label='Mean and 95% CI' if i == 0 else None)
        ax.axvline(0, color='0.25', linewidth=0.8, linestyle='--')
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('Return difference', fontsize=8.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_yticks(range(3), LABELS)
        ax.grid(axis='x', color='0.90', linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.01), fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=0.8)
    fig.savefig(HERE / 'history_effects.pdf', bbox_inches='tight')
    fig.savefig(HERE / 'history_effects.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    table = []
    for cond, label in zip(CONDITIONS, LABELS):
        e = results[cond]
        a, n, r = [e[k]['mean_ci95'][0] for k in ['adaptive', 'no_history', 'reactive']]
        table.append(f'{label} & {a:.1f} & {n:.1f} & {r:.1f} \\\\')
    (HERE / 'return_rows.tex').write_text('\\newcommand{\\ReturnRows}{%\n'+'\n'.join(table)+'\n}\n')
    table = []
    for cond, label in zip(CONDITIONS, LABELS):
        e = results[cond]
        g, gl, gh = e['G']['mean_ci95']
        d, dl, dh = e['D']['mean_ci95']
        table.append(f'{label} & ${g:+.1f}$ & $[{gl:+.1f}, {gh:+.1f}]$ & '
                     f'${d:+.1f}$ & $[{dl:+.1f}, {dh:+.1f}]$ \\\\')
    (HERE / 'effect_rows.tex').write_text('\\newcommand{\\EffectRows}{%\n'+'\n'.join(table)+'\n}\n')
    print(json.dumps({c: {k: results[c][k]['mean_ci95'] for k in ['G', 'D']}
                      for c in CONDITIONS}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Optional original fault_benchmark_results directory')
    main(parser.parse_args().source)
