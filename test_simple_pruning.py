#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")

from scoredrulesets.estimators.tree_transform import TreeTransformParams, _aggressive_atom_pruning
from scoredrulesets.schema import Rule, Atom

print("✓ Import erfolgreich")

# Teste die Pruning-Funktion mit einfachen Test-Regeln
test_rules = [
    Rule(atoms=[], scores=[1.0, 0.0, 0.0], rule_id="default"),
    Rule(atoms=[
        Atom(feature="x1", op="<=", value=5.0),
        Atom(feature="x2", op=">", value=3.0),
    ], scores=[0.5, 0.0, 0.0], rule_id="r1"),
    Rule(atoms=[
        Atom(feature="x3", op="<=", value=2.0),
        Atom(feature="x4", op=">", value=1.0),
        Atom(feature="x5", op="<=", value=10.0),
    ], scores=[0.0, 1.0, 0.0], rule_id="r2"),
]

print(f"\nVor Pruning:")
for r in test_rules:
    print(f"  {r.rule_id}: {len(r.atoms)} atoms")

pruned_rules = _aggressive_atom_pruning(test_rules, prune_lambda=2.0)

print(f"\nNach Pruning (λ=2.0):")
total_atoms_before = sum(len(r.atoms) for r in test_rules)
total_atoms_after = sum(len(r.atoms) for r in pruned_rules)
for r in pruned_rules:
    print(f"  {r.rule_id}: {len(r.atoms)} atoms")
print(f"\nAtome entfernt: {total_atoms_before - total_atoms_after}")

