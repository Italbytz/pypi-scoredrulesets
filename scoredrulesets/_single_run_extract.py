from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets import ScoredRuleSetClassifier, format_ruleset_table
from scoredrulesets.benchmarking.datasets import load_dataset_registry


def main() -> None:
    registry = load_dataset_registry(include_online_uci=False, include_synthetic=True)
    bundle = registry["synth_monk3"]
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.25, random_state=42, stratify=bundle.y
    )

    models = {
        "ruleGP": ScoredRuleSetClassifier(
            backend="rulegp",
            backend_params={
                "population_size": 150,
                "generations": 250,
                "max_rules": 12,
                "max_atoms_per_rule": 5,
                "tournament_size": 4,
                "early_stopping_rounds": 30,
                "max_fit_seconds": 240,
                "enable_compaction": True,
            },
            random_state=42,
        ),
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

    results = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        rs = clf.ruleset_
        n_rules = len(rs.rules)
        n_atoms = sum(len(rule.atoms) for rule in rs.rules)
        results.append((name, f1, n_rules, n_atoms, clf))

    print("DATASET=synth_monk3")
    for name, f1, n_rules, n_atoms, _ in sorted(results, key=lambda row: row[1], reverse=True):
        print(f"{name:8s} F1_macro={f1:.4f} rules={n_rules} atoms={n_atoms}")

    best = sorted(results, key=lambda row: row[1], reverse=True)[0]
    print(f"\nBEST_MODEL={best[0]}")
    print("RULESET_TABLE_START")
    print(format_ruleset_table(best[4].ruleset_))
    print("RULESET_TABLE_END")


if __name__ == "__main__":
    main()
