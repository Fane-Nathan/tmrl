#!/usr/bin/env python3
"""Build and execute a compact, reproducible data-quality notebook."""
import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        md("## tl;dr\n\nThe dataset consists of two finish-confirmed, assisted runs on one known map. "
           "One entire lap is held out. This audit checks causal labels and provenance; it does not "
           "establish learned driving, collision-free driving, or one-shot generalization."),
        md("## Context & Methods\n\nGrain: one camera observation followed by its applied command. "
           "Validate timestamps, finite/ranged values, terminal flags, episode-local frame history, "
           "input hashes, and split isolation.\n\n### Key Assumptions\n\nCamera labels are user-declared. "
           "Track identity uses the configured map-file hash plus live start position, not a live map UID. "
           "Local receive times cannot prove exact GPU-frame/telemetry synchronization. "
           "The original reference contains post-finish motion, so its final index is not a finish metric."),
        md("## Data\n\nOnly the explicit dataset and its two listed source runs are read; no broad filesystem scan."),
        code(f"from pathlib import Path\nimport json, sys\nimport numpy as np\nimport torch\n"
             f"ROOT = Path({str(root)!r})\nsys.path.insert(0, str(ROOT))\n"
             "from scripts.export_live_driving_demos import load_verified_run, sha256\n"
             f"DATASET = Path({str(args.dataset.resolve())!r})\n"
             "manifest = json.loads(DATASET.with_suffix('.metadata.json').read_text(encoding='utf-8'))\n"
             "assert sha256(DATASET) == manifest['dataset_sha256']\n"
             "payload = torch.load(DATASET, map_location='cpu', weights_only=False)\n"
             "episodes = payload['episodes']\n"
             "print({'dataset': str(DATASET), 'episodes': len(episodes), 'steps': sum(len(e['actions']) for e in episodes)})"),
        md("## Results\n\n### 1. Re-run causal-source and provenance checks"),
        code("rows = []\nfor episode, saved_source in zip(episodes, manifest['sources']):\n"
             "    reconstructed, source = load_verified_run(saved_source['result_path'])\n"
             "    assert source['archive_sha256'] == saved_source['archive_sha256']\n"
             "    assert source['result_sha256'] == saved_source['result_sha256']\n"
             "    for key in ('imgs', 'states', 'actions', 'previous_actions', 'terminated'):\n"
             "        np.testing.assert_array_equal(episode[key], reconstructed[key])\n"
             "    rows.append({key: source[key] for key in ('run_id', 'steps', 'finished', 'max_deviation_m', 'sampling_interval_max_ms')})\n"
             "print(json.dumps(rows, indent=2))"),
        md("### 2. Check split isolation and data limitations"),
        code("assert len(episodes) == len(manifest['sources']) == manifest['episode_count']\n"
             "assert sum(len(ep['actions']) for ep in episodes) == manifest['transition_count']\n"
             "train = {ep['metadata']['archive_sha256'] for ep in episodes if ep['split'] == 'train'}\n"
             "validation = {ep['metadata']['archive_sha256'] for ep in episodes if ep['split'] == 'validation'}\n"
             "assert train and validation and train.isdisjoint(validation)\n"
             "assert len({ep['metadata']['map_sha256'] for ep in episodes}) == 1\n"
             "print({'train_laps': len(train), 'validation_laps': len(validation), 'maps': 1, 'audit': 'PASS'})\n"
             "print('Limitations:')\nfor item in manifest['use_limitations']:\n    print('- ' + item)"),
        md("## Takeaways\n\nUse these as known-track teacher labels with continuous steering. "
           "Mask diagnostic GPS and current-control fields before training. Preserve the held-out lap. "
           "Keep weights as candidates until independent closed-loop runs pass; the existing learned policy "
           "failed live even though its saved-frame imitation audit passed."),
    ])
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    try:
        NotebookClient(notebook, km=manager, timeout=60,
                       resources={"metadata": {"path": str(root)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.notebook)
    print(f"Executed audit: {args.notebook.resolve()}")


if __name__ == "__main__":
    main()
