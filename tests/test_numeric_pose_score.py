#!/usr/bin/env python3
"""CPU-only regression tests for quasar.numeric_pose_score."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


from quasar import numeric_pose_score as analysis
from quasar import score_vsibench_official_like_predictions as scorer


def source_row(sample_id: str, *, dataset: str = "vsi", qtype: str | None = None) -> dict:
    if dataset == "vsi":
        canonical = qtype or "object_abs_distance"
        return {
            "sample_id": sample_id,
            "source": {"tag": "official_eval_legacy_vsi"},
            "question": {"canonical_question_type": canonical},
            "answer": {"type": "numeric", "numeric_value": 10.0, "text": "10"},
            "scene": {"video_id": "scene_1"},
        }
    canonical = qtype or "vsti_camera_movement_direction"
    return {
        "sample_id": sample_id,
        "source": {"tag": "vsti_official_row32"},
        "question": {"canonical_question_type": canonical},
        "answer": {
            "type": "multiple_choice",
            "option_letter": "B",
            "options": [
                {"letter": "A", "text": "left"},
                {"letter": "B", "text": "right"},
            ],
        },
    }


def raw_row(sample_id: str, *, dataset: str = "vsi", arm: str = "hard_off") -> dict:
    source = source_row(sample_id, dataset=dataset)
    return {
        "sample_id": sample_id,
        "bench": "VSI" if dataset == "vsi" else "VSTI",
        "source_tag": source["source"]["tag"],
        "canonical_question_type": source["question"]["canonical_question_type"],
        "prediction_text": "10" if dataset == "vsi" else "B",
        "prediction_text_raw": "10" if dataset == "vsi" else "B",
        "answer_type": source["answer"]["type"],
        "query_pose_protocol": {
            "mode": "legacy" if dataset == "vsi" else "provided_full",
            "effective_pose_input_mask": False if arm == "hard_off" else True,
            "geometry_gauge": "scene_1",
        },
        "gate1": {
            "arm": arm,
            "effective_reference_type_id": 2,
            "query_pose_vec12_all_zero": arm == "hard_off",
            "descriptor_query_channels_all_zero": arm == "hard_off",
        },
        "intervention_audit": {
            "pose_projection_after_mask_max_abs": 0.0 if arm == "hard_off" else 0.25,
            "visual_writeback_max_abs": 0.5,
            "deepstack_layers": 3,
            "spatial_query_count": 4,
        },
    }


def scored_row(sample_id: str, qtype: str, score: float, cluster: str) -> dict:
    return {
        "sample_id": sample_id,
        "canonical_question_type": qtype,
        "task_group": qtype,
        "subgroup": "synthetic",
        "scene_cluster": cluster,
        "score": score,
    }


class FullAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = scorer
        cls.scorer_path = Path(scorer.__file__)

    def test_exact_ids_reject_missing_duplicate_and_reorder(self) -> None:
        expected = ["a", "b", "c"]
        analysis.validate_prediction_ids(["a", "b", "c"], expected, "ok")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.validate_prediction_ids(["a", "a", "c"], expected, "duplicate")
        with self.assertRaisesRegex(ValueError, "missing"):
            analysis.validate_prediction_ids(["a", "b"], expected, "missing")
        with self.assertRaisesRegex(ValueError, "order"):
            analysis.validate_prediction_ids(["b", "a", "c"], expected, "reordered")

    def test_hard_off_rejects_missing_mask_and_label_leak(self) -> None:
        source = source_row("vsi_1")
        raw = raw_row("vsi_1")
        analysis.validate_raw_prediction(raw, source, dataset="vsi", arm="hard_off")
        analysis.validate_raw_prediction({**raw, "target_text": ""}, source, dataset="vsi", arm="hard_off")
        missing_mask = {**raw, "query_pose_protocol": dict(raw["query_pose_protocol"])}
        del missing_mask["query_pose_protocol"]["effective_pose_input_mask"]
        with self.assertRaisesRegex(KeyError, "effective_pose_input_mask"):
            analysis.validate_raw_prediction(missing_mask, source, dataset="vsi", arm="hard_off")
        leaked = {**raw, "target_text": "10"}
        with self.assertRaisesRegex(ValueError, "leaks"):
            analysis.validate_raw_prediction(leaked, source, dataset="vsi", arm="hard_off")

    def test_frozen_score_methods_for_vsi_and_vsti(self) -> None:
        vsi_raw = raw_row("vsi_1", dataset="vsi")
        vsi = analysis.score_one(
            vsi_raw,
            source_row("vsi_1", dataset="vsi"),
            dataset="vsi",
            arm="hard_off",
            subgroup="semantic_1162",
            preflight=False,
            scorer=self.scorer,
        )
        self.assertEqual(vsi["metric"], "MRA:.5:.95:.05")
        self.assertEqual(vsi["score"], 1.0)

        vsti_raw = raw_row("vsti_1", dataset="vsti")
        vsti = analysis.score_one(
            vsti_raw,
            source_row("vsti_1", dataset="vsti"),
            dataset="vsti",
            arm="hard_off",
            subgroup="legacy_5598",
            preflight=True,
            scorer=self.scorer,
        )
        self.assertEqual(vsti["metric"], "choice_accuracy")
        self.assertEqual(vsti["score"], 1.0)
        self.assertEqual(vsti["task_group"], "camera_movement_direction")
        self.assertTrue(self.scorer_path.name.endswith("score_vsibench_official_like_predictions.py"))

        numeric_source = source_row(
            "vsti_2", dataset="vsti", qtype="vsti_camera_displacement"
        )
        numeric_source["answer"] = {"type": "numeric", "numeric_value": 10.0, "text": "10 meters"}
        numeric_raw = raw_row("vsti_2", dataset="vsti")
        numeric_raw.update(
            canonical_question_type="vsti_camera_displacement",
            answer_type="numeric",
            prediction_text="The answer is 10 meters",
            prediction_text_raw="The answer is 10 meters",
        )
        numeric = analysis.score_one(
            numeric_raw,
            numeric_source,
            dataset="vsti",
            arm="hard_off",
            subgroup="legacy_5598",
            preflight=False,
            scorer=self.scorer,
        )
        self.assertEqual(numeric["metric"], "MRA:.5:.95:.05")
        self.assertEqual(numeric["score"], 1.0)

    def test_paired_direction_is_hard_off_minus_current(self) -> None:
        current = [
            scored_row("a", "object_counting", 0.0, "s1"),
            scored_row("b", "object_counting", 0.0, "s2"),
        ]
        hard_off = [
            scored_row("a", "object_counting", 1.0, "s1"),
            scored_row("b", "object_counting", 1.0, "s2"),
        ]
        result = analysis.paired_bootstrap(
            current,
            hard_off,
            dataset="vsi",
            scorer=self.scorer,
            group_kind="question_type",
            group="object_counting",
            repetitions=200,
            seed=analysis.BOOTSTRAP_SEED,
        )
        interval = result["intervals"]["primary_score"]
        self.assertEqual(result["direction"], "hard_off-current")
        self.assertEqual(interval["observed_delta"], 1.0)
        self.assertEqual(interval["ci95"], [1.0, 1.0])
        self.assertEqual(interval["direction_significant"], "positive")

    def test_scene_bootstrap_marks_absent_qtype_unavailable(self) -> None:
        current = [
            scored_row("a", "object_counting", 0.0, "only_count"),
            scored_row("b", "object_abs_distance", 0.0, "only_distance"),
        ]
        hard_off = [
            scored_row("a", "object_counting", 1.0, "only_count"),
            scored_row("b", "object_abs_distance", 1.0, "only_distance"),
        ]
        result = analysis.paired_bootstrap(
            current,
            hard_off,
            dataset="vsi",
            scorer=self.scorer,
            group_kind="subgroup",
            group="synthetic",
            repetitions=200,
            seed=analysis.BOOTSTRAP_SEED,
        )
        self.assertEqual(result["intervals"]["qtype::object_counting"]["status"], "unavailable")
        self.assertGreater(
            result["intervals"]["qtype::object_counting"]["unavailable_repetitions"], 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
