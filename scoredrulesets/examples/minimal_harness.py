from pathlib import Path

from sklearn.datasets import load_iris

from scoredrulesets import ScoredRuleSetClassifier, dump_ruleset_json, import_logicgp_json


def run_training_demo() -> None:
    X, y = load_iris(return_X_y=True)

    clf = ScoredRuleSetClassifier(
        backend="cart",  # Fuer reproduzierbares Beispiel ohne Zusatzabhaengigkeiten
        backend_params={"max_depth": 3},
        transform_params={"depth_decay_lambda": 2.0},
        random_state=42,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    output_file = Path("iris_ruleset.json")
    dump_ruleset_json(ruleset, output_file)
    print(f"Gespeichert: {output_file} ({len(ruleset.rules)} Regeln)")


def run_logicgp_import_demo() -> None:
    logicgp_path = Path("ext/logicgp_model.json")
    if not logicgp_path.exists():
        print("logicGP-Beispieldatei nicht gefunden, ueberspringe Import-Demo")
        return

    ruleset = import_logicgp_json(logicgp_path)
    output_file = Path("logicgp_converted_ruleset.json")
    dump_ruleset_json(ruleset, output_file)
    print(f"Konvertiert: {logicgp_path} -> {output_file}")


if __name__ == "__main__":
    run_training_demo()
    run_logicgp_import_demo()

