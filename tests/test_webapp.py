"""Unit tests for the helper functions of the web application (no training needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from webapp import (  # noqa: E402
    dataset_meta, detect_stage, parse_result_line, registered_name,
)


def test_registered_name_matches_training_rule():
    assert registered_name("wine") == "wine-classifier-prod"
    assert registered_name("iris") == "iris-classifier-prod"


def test_parse_result_line_reads_summary():
    line = 'RESULT_JSON:{"version": 3, "promoted": true}'
    assert parse_result_line(line) == {"version": 3, "promoted": True}


def test_parse_result_line_ignores_other_lines():
    assert parse_result_line("2026-01-01 [INFO] Loading raw data...") is None
    assert parse_result_line("RESULT_JSON:{broken") is None


def test_detect_stage_only_moves_forward():
    stage = "preprocess"
    stage = detect_stage("[INFO] Augmenting training data (factor=2)", stage)
    assert stage == "augment"
    stage = detect_stage("[INFO] [random_forest] metrics: {...}", stage)
    assert stage == "train"
    # An earlier marker must not move the stage backwards.
    assert detect_stage("[INFO] Loading raw data...", stage) == "train"
    assert detect_stage("[INFO] Promoted v1 to Production", stage) == "gate"


def test_dataset_meta_wine_and_iris():
    wine = dataset_meta("wine")
    assert wine["n_features"] == 13
    assert len(wine["targets"]) == 3
    assert len(wine["examples"]) == 3
    assert len(wine["examples"][0]["values"]) == 13

    iris = dataset_meta("iris")
    assert iris["n_features"] == 4
    assert iris["targets"] == ["setosa", "versicolor", "virginica"]
