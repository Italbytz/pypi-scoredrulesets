from __future__ import annotations

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import ScoredRuleSetClassifier
from scoredrulesets.benchmarking.datasets import load_dataset_registry


def _rule_to_text(rule, max_atoms: int = 3) -> str:
    atoms = getattr(rule, "atoms", []) or []
    if not atoms:
        return "DEFAULT"
    chunks = []
    for atom in atoms[:max_atoms]:
        chunks.append(f"{atom.feature} {atom.op} {atom.value}")
    if len(atoms) > max_atoms:
        chunks.append("...")
    return " AND ".join(chunks)


def main() -> None:
    registry = load_dataset_registry(include_online_uci=False, include_synthetic=True)
    bundle = registry["synth_checkerboard_4x4"]

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.25,
        random_state=42,
        stratify=bundle.y,
    )

    models = {
        "ruleLCS": ScoredRuleSetClassifier(
            backend="rulelcs",
            backend_params={
                "population_size": 200,
                "n_iterations": 50,
                "n_repetitions": 2,
                "tournament_size": 4,
                "crossover_prob": 0.6,
                "mutation_prob": 0.6,
                "max_rules": 15,
                "default_class_policy": "major",
            },
            random_state=42,
        ),
        "HS": ScoredRuleSetClassifier(
            backend="hs",
            transform_params={"prune_atoms": True},
            random_state=42,
        ),
    }

    summary: list[tuple[str, float, int, int, object]] = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        rs = clf.ruleset_
        n_rules = len(rs.rules)
        n_atoms = sum(len(r.atoms) for r in rs.rules)
        summary.append((name, f1, n_rules, n_atoms, rs))

    print("DATASET=synth_checkerboard_4x4")
    for name, f1, n_rules, n_atoms, _ in sorted(summary, key=lambda row: row[1], reverse=True):
        print(f"{name:7s} F1_macro={f1:.4f} rules={n_rules} atoms={n_atoms}")

    best = sorted(summary, key=lambda row: row[1], reverse=True)[0]
    best_name, best_f1, _, _, best_rs = best
    print(f"\nBEST_MODEL={best_name} (F1_macro={best_f1:.4f})")

    non_default = [r for r in best_rs.rules if getattr(r, "atoms", None)]
    default = [r for r in best_rs.rules if not getattr(r, "atoms", None)]

    print("RULE_EXAMPLE_START")
    for idx, rule in enumerate(non_default[:3], start=1):
        print(f"R{idx}: {_rule_to_text(rule)}")
        print(f"    scores={list(rule.scores)}")
    if default:
        print("R_default: DEFAULT")
        print(f"    scores={list(default[0].scores)}")
    print("RULE_EXAMPLE_END")


if __name__ == "__main__":
    main()
