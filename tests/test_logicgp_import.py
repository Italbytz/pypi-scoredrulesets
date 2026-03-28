from pathlib import Path

from scoredrulesets import import_logicgp_json


LOGICGP_FILE = Path(__file__).resolve().parent / "fixtures" / "logicgp_model.json"


def test_logicgp_import_basic():
    ruleset = import_logicgp_json(LOGICGP_FILE)
    assert ruleset.metadata["source"] == "logicgp"
    assert len(ruleset.class_labels) == 3
    assert len(ruleset.rules) == 3
    assert ruleset.rules[1].atoms[0].op == "in"

