import pytest

from quasar.retained_predictions_score import aggregate, rescore_row


def test_numeric_rescoring_uses_answer_and_first_token():
    gold = {"sample_id": "synthetic", "question": {"canonical_question_type": "object_counting"},
            "answer": {"type": "count", "text": "4", "numeric_value": 4}}
    row = {"sample_id": "synthetic", "prediction_text": "2", "score": 1.}
    result = rescore_row(row, gold, "VSI")
    assert result["score"] == .1
    assert result["metric"] == "MRA:.5:.95:.05"


def test_choice_rescoring_does_not_trust_retained_score():
    gold = {"sample_id": "synthetic", "question": {"canonical_question_type": "vsti_camera_obj_rel_dist_v1"},
            "answer": {"type": "multiple_choice", "text": "table", "option_letter": "A",
                       "options": [{"letter": "A", "text": "table"}, {"letter": "B", "text": "chair"}]}}
    row = {"sample_id": "synthetic", "prediction_text": "B. chair", "score": 1.}
    assert rescore_row(row, gold, "VSTI")["score"] == 0.


def test_task_macro_differs_from_row_weighting_and_rejects_duplicate():
    records = [{"sample_id": str(i), "canonical_question_type": "vsti_camera_obj_rel_dist_v1", "score": 1.} for i in range(3)]
    records += [{"sample_id": "3", "canonical_question_type": "vsti_camera_displacement", "score": 0.}]
    assert aggregate(records, "vsti_5task")["score_fraction"] == .5
    assert aggregate(records, "exact_micro")["score_fraction"] == .75
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(records + [records[0]], "vsti_5task")
