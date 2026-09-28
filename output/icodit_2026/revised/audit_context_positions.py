"""Compare checkpoint position embeddings with seeded initialization.

Run from the original workspace using its research Python environment:
  python output/icodit_2026/revised/audit_context_positions.py --workspace .
This reads checkpoints and constructs untrained models. It does not train,
evaluate driving, or alter any saved checkpoint.
"""
from pathlib import Path
import argparse
import json
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, required=True)
args = parser.parse_args()
root = args.workspace.resolve()
sys.path.insert(0, str(root))
from scripts.train_transformer_fault_adaptation import TransformerRL2Agent

rows = []
for seed in range(42, 50):
    torch.manual_seed(seed)
    initial = TransformerRL2Agent().trunk.pos_emb.detach().clone()
    path = root / 'fault_benchmark_results' / f'transformer_seed_{seed}' / 'best_validation_checkpoint.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    actual = state['trunk.pos_emb']
    rows.append(dict(seed=seed, position_shape=list(actual.shape),
                     max_change_positions_0_to_31=float((actual[:, :32]-initial[:, :32]).abs().max()),
                     max_change_positions_32_to_127=float((actual[:, 32:]-initial[:, 32:]).abs().max())))
result = dict(interpretation='Compares archived checkpoint positional embeddings with fresh seeded initialization of the current local implementation. Positions 32 onward are outside the training windows; history inference uses positions through 63. The benchmark files were untracked at the recorded commit. This is a consistency diagnostic, not proof of historical source identity.', runs=rows)
output = Path(__file__).resolve().parent / 'data' / 'context_position_audit.json'
output.write_text(json.dumps(result, indent=2))
print(json.dumps(rows, indent=2))
