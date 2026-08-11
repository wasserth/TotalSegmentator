import numpy as np
import pytest

from totalsegmentator.cnn import (
    CNN_MODEL_TARGET_ORDERS,
    _apply_regression_target_denormalization,
    _format_all_body_stats,
    _format_regression_result,
    _get_model_target_names,
    _get_nr_classes,
    _validate_modality_and_target,
)


def test_new_model_target_orders_are_modality_specific():
    assert len(CNN_MODEL_TARGET_ORDERS["mr"]) == 10
    assert CNN_MODEL_TARGET_ORDERS["mr"][-1] == "mr_sequence_int"

    assert len(CNN_MODEL_TARGET_ORDERS["ct"]) == 13
    assert "KVP" in CNN_MODEL_TARGET_ORDERS["ct"]
    assert "mr_sequence_int" not in CNN_MODEL_TARGET_ORDERS["ct"]


@pytest.mark.parametrize("modality", ["mr", "ct"])
def test_model_target_order_falls_back_by_modality(modality):
    expected = CNN_MODEL_TARGET_ORDERS[modality]
    assert _get_model_target_names({}, len(expected), modality) == expected


def test_checkpoint_target_count_must_match_model_output_count():
    with pytest.raises(ValueError, match="defines 2 target names"):
        _get_model_target_names(
            {"reg_target_names": ["PatientWeight", "PatientSize"]},
            output_count=3,
            modality="mr",
        )


def test_nr_classes_falls_back_to_checkpoint_target_names():
    hparams = {"reg_target_names": CNN_MODEL_TARGET_ORDERS["ct"]}
    assert _get_nr_classes(hparams) == 13


def test_normalized_huber_outputs_are_denormalized():
    pred = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    hparams = {
        "loss": "mse_huber",
        "reg_target_normalize": True,
        "reg_target_mean": [10.0, 20.0, 30.0],
        "reg_target_std": [2.0, 3.0, 4.0],
    }

    np.testing.assert_allclose(
        _apply_regression_target_denormalization(pred, hparams),
        [10.0, 23.0, 26.0],
    )


@pytest.mark.parametrize(
    ("modality", "expected_targets"),
    [
        (
            "mr",
            [
                "weight",
                "size",
                "age",
                "sex",
                "contrast",
                "verte_upper",
                "verte_lower",
                "noise",
                "mr_sequence",
            ],
        ),
        (
            "ct",
            [
                "weight",
                "size",
                "age",
                "sex",
                "manufacturer",
                "kvp",
                "xray_tube_current",
                "convolution_kernel",
                "contrast",
                "pi_time",
                "verte_upper",
                "verte_lower",
                "noise",
            ],
        ),
    ],
)
def test_all_modality_targets_are_formatted(modality, expected_targets):
    training_names = CNN_MODEL_TARGET_ORDERS[modality]
    preds = np.arange(len(training_names), dtype=np.float32)[None, :]
    hparams = {"reg_target_names": training_names}

    result = _format_all_body_stats(preds, hparams, modality)

    assert list(result) == expected_targets


def test_target_must_be_available_for_modality():
    with pytest.raises(ValueError, match="for modality 'mr'"):
        _validate_modality_and_target("mr", "kvp")


@pytest.mark.parametrize(
    ("target", "score", "expected"),
    [
        ("contrast", 0.49, "no"),
        ("contrast", 0.5, "yes"),
        ("manufacturer", 1.6, "ge"),
        ("manufacturer", 20.0, "other"),
        ("mr_sequence", 3.6, "stir"),
        ("mr_sequence", 20.0, "other"),
        ("verte_upper", 8.2, "T1"),
        ("verte_lower", 24.0, "L5"),
        ("verte_lower", 0.0, "C1"),
        ("verte_lower", -1.0, "C1"),
        ("verte_lower", -2.0, None),
        ("verte_upper", 25.0, "L5"),
        ("verte_upper", 26.0, "L5"),
        ("verte_upper", 27.0, None),
    ],
)
def test_categorical_targets_are_mapped(target, score, expected):
    result = _format_regression_result(
        np.array([score], dtype=np.float32), target
    )

    assert result["value"] == expected
