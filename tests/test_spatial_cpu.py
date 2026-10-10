"""合成输入检查：读取、坐标、历史目标优先级，以及进入语言模型的两条写回路径。"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from quasar import train_native_qwen3vl2b_glow_patch_spatial as basic
from quasar import train_native_qwen3vl2b_glow_patch_spatial_moe_metric as spatial
from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import NativeQwenGlowPatchSpatialDecoderAuxModel
from quasar import train_native_qwen3vl2b_glow_patch_spatial_decoder_aux as decoder


@pytest.fixture
def sample():
    points = torch.tensor([[[[0., 0., 0.], [2., 0., 0.]], [[0., 0., 4.], [2., 0., 4.]]]])
    return spatial.NativeGlowSample(
        sample_id="synthetic-0", source_tag="synthetic", scene_id="room", video_id="clip",
        question="synthetic", canonical_question_type="object_size_estimation", answer_type="numeric",
        target_text="1", prompt="synthetic", image_paths=("synthetic.png",),
        points=points, patch_valid=torch.ones((1, 2, 2), dtype=torch.bool),
        patch_confidence=torch.ones((1, 2, 2)), source_poses=torch.eye(4).unsqueeze(0),
        reference_type_id=spatial.reference_type_mapping("simple3")["target_camera_pose"],
        query_pose=torch.eye(4), query_pose_valid=True,
    )


def test_real_npz_reader_preserves_patch_order_and_masks(tmp_path):
    points = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    points[1, 2, 0] = np.nan
    mask = np.ones((2, 4), dtype=bool)
    mask[0, 1] = False
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = 7.
    path = tmp_path / "synthetic.npz"
    np.savez(path, frame_image_paths=np.array(["a.png", "b.png"]), target_points=points,
             camera_pose=poses, patch_grid_hw=np.array([[2, 2], [2, 2]]),
             token_mask=mask, token_confidence=np.full((2, 4), .75, dtype=np.float32))
    dataset = spatial.NativeAnswerPatchSpatialDataset(rows=[{"sample_id": "synthetic"}],
        max_visual_frames=2, scannet_frame_root=None, depth_grid_hw=(2, 2), reference_schema="simple3")
    paths, actual, valid, confidence, actual_poses = dataset._load_cut3r_npz(sample_id="synthetic", path=path)
    assert paths == ["a.png", "b.png"]
    torch.testing.assert_close(actual[0].reshape(4, 3), torch.from_numpy(points[0]))
    assert not valid[0, 0, 1] and not valid[1, 1, 0]
    assert torch.isfinite(actual).all()
    assert confidence[1, 1, 1] == .75 and actual_poses[1, 0, 3] == 7.
    # 错误网格应明确拒绝，避免把视觉位置悄悄对错。
    np.savez(path, frame_image_paths=np.array(["a.png", "b.png"]), target_points=points,
             camera_pose=poses, patch_grid_hw=np.array([[2, 2], [1, 4]]))
    with pytest.raises(ValueError, match="variable CUT3R grid"):
        dataset._load_cut3r_npz(sample_id="synthetic", path=path)


def test_geometry_scale_camera_and_bev():
    points = torch.tensor([[0., 0., 0.], [4., 2., 8.], [100., 100., 100.]])
    center, scale = spatial.compute_scene_center_scale(points=points, valid=torch.tensor([True, True, False]))
    torch.testing.assert_close(center, torch.tensor([2., 1., 4.]))
    assert scale == 4.
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    pose[:3, 3] = torch.tensor([1., 0., 2.])
    out = spatial.transform_world_points_to_camera(points=torch.tensor([[[3., 0., 2.]]]), camera_pose=pose, scene_scale=torch.tensor(2.))
    torch.testing.assert_close(out, torch.tensor([[[0., 0., 1.]]]))
    bev = spatial.build_bev_coord_targets(points=points[:2].unsqueeze(0), center=center, scale=scale)
    torch.testing.assert_close(bev, torch.tensor([[[-.5, -1.], [.5, 1.]]]))


def test_dataset_reads_full_native_schema_and_cached_metric_targets(tmp_path, sample):
    paths = []
    for index in range(2):
        path = tmp_path / f"frame{index}.png"
        Image.new("RGB", (4, 4), color=(index * 80, 0, 0)).save(path)
        paths.append(str(path))
    points = torch.cat([sample.points, sample.points + 10.], dim=0)
    geometry = tmp_path / "geometry.npz"
    np.savez(geometry, frame_image_paths=np.array(paths), target_points=points.numpy().reshape(2, 4, 3),
             camera_pose=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
             patch_grid_hw=np.array([[2, 2], [2, 2]]), token_mask=np.ones((2, 4), dtype=bool))
    pose = torch.eye(4)
    pose[0, 3] = 3.
    row = {"sample_id": "synthetic-full", "source": {"tag": "synthetic"},
           "scene": {"scene_id": "room", "video_id": "clip"},
           "question": {"text": "What is the size?", "canonical_question_type": "object_size_estimation",
                        "reference_condition": "target_camera_pose"},
           "answer": {"type": "numeric", "text": "2", "numeric_value": 2.}, "inputs": {},
           "refs": {"geometry": {"cut3r_npz_path": str(geometry)}, "pose": {}, "simple3_aux": {
               "reference_schema": "simple3", "reference_type": "target_camera_pose", "reference_type_id": 0,
               "query_pose_valid": True, "query_pose_4x4": pose.tolist(), "target_camera": {"enabled": True},
               "metric_aux": {"enabled": True, "scene": {"log1p_scene_scale": 9., "log1p_room_extent_xyz": [8., 7., 6.]},
                   "camera": {"valid": True, "log1p_primary_first_max": [5., 4., 3.]},
                   "objects": {"valid": True, "log1p_mean_size_xyz": [2., 3., 4.], "log1p_count": 7.,
                               "pair_distance": {"valid": True, "log1p_min_mean_max": [1., 2., 3.]},
                               "camera_object_distance": {"valid": True, "log1p_min_mean_max": [4., 5., 6.]}}}}}}
    dataset = spatial.NativeAnswerPatchSpatialDataset(rows=[row], max_visual_frames=2, scannet_frame_root=None,
        depth_grid_hw=(2, 2), reference_schema="simple3", simple3_aux_required=True)
    loaded = dataset[0]
    assert loaded.image_paths == tuple(paths) and loaded.target_text == "2"
    torch.testing.assert_close(loaded.points, points)
    torch.testing.assert_close(loaded.query_pose, pose)
    assert loaded.aux_metric_scene_scale.item() == 9.
    assert sum(getattr(loaded, "aux_metric_" + key).numel() for key in
               ["scene_scale", "room_extent", "camera_displacement", "object_size", "object_pair_distance", "camera_object_distance", "object_count"]) == 17


@pytest.mark.parametrize("reference", ["target_camera_pose", "semantic_query_pose", "canonical_bev_scene_pose"])
def test_explicit_reference_reader(reference):
    mapping = spatial.reference_type_mapping("simple3")
    aux = {"reference_schema": "simple3", "reference_type": reference,
           "reference_type_id": mapping[reference], "query_pose_valid": True,
           "query_pose_4x4": torch.eye(4).tolist(), "target_camera": {"enabled": True},
           "semantic_query": {"enabled": True},
           "canonical_bev": {"enabled": True, "scene_center": [1., 2., 3.], "scene_scale": 4.,
                             "object_center_supervision": True,
                             "object_centers": [{"center_world": [2., 3., 4.], "size_world": [1., 1., 1.]}]}}
    result = spatial.resolve_reference_condition_from_simple3_aux(row={}, sample_id="synthetic", simple3_aux=aux, reference_schema="simple3")
    assert result[0] == mapping[reference] and result[2]
    torch.testing.assert_close(result[1], torch.eye(4))
    if reference == "canonical_bev_scene_pose":
        torch.testing.assert_close(result[3], torch.tensor([1., 2., 3.]))
        assert result[4] == 4.
    with pytest.raises(ValueError, match="reference_type_id mismatch"):
        spatial.resolve_reference_condition_from_simple3_aux(row={}, sample_id="synthetic",
            simple3_aux={**aux, "reference_type_id": 99}, reference_schema="simple3")


def test_collator_order_cached_targets_and_pose_mask(sample):
    collator = spatial.NativeGlowPatchSpatialCollator(processor=SimpleNamespace(tokenizer=None),
        max_pixels=64, spatial_merge_size=2, bev_grid_size=4, reference_schema="simple3")
    values = {"scene_scale": [9.], "room_extent": [8., 7., 6.], "camera_displacement": [5., 4., 3.],
              "object_size": [2., 3., 4.], "object_pair_distance": [1., 2., 3.],
              "camera_object_distance": [4., 5., 6.], "object_count": [7.]}
    cached = replace(sample, **{"aux_metric_" + key: torch.tensor(value) for key, value in values.items()})
    second = replace(cached, sample_id="synthetic-1", points=sample.points + 10.)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    batch = collator._build_spatial_payload(samples=[cached, second], image_grid_thw=grid)
    torch.testing.assert_close(batch["spatial_points"], torch.cat([sample.points.reshape(4, 3), second.points.reshape(4, 3)]))
    assert batch["spatial_token_sample_ids"].tolist() == [0] * 4 + [1] * 4
    assert batch["spatial_descriptors"].shape == (8, spatial.SPATIAL_DESCRIPTOR_DIM)
    for key, value in values.items():
        torch.testing.assert_close(batch["metric_" + key + "_targets"][0], torch.tensor(value))
    missing = replace(cached, effective_pose_input_mask=False, query_pose_valid=False)
    off = collator._build_spatial_payload(samples=[missing], image_grid_thw=grid[:1])
    assert torch.count_nonzero(off["query_pose_vec12"]) == 0
    assert torch.count_nonzero(off["query_frame_coord_targets"]) == 0
    assert off["metric_camera_displacement_valid"].item() == 0
    assert off["metric_camera_object_distance_valid"].item() == 0
    assert off["metric_object_count_valid"].item() == 1


@pytest.fixture
def tiny_qwen():
    # 随机小模型只检查接口；不读取预训练权重，不生成答案。
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "max_position_embeddings": 128,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 500000.,
                                         "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
        vision_config={"depth": 3, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "patch_size": 2, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "out_hidden_size": 32, "deepstack_visual_indexes": [0, 1]},
        image_token_id=1, vision_start_token_id=2, vision_end_token_id=3, video_token_id=10,
    )
    config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize("use_cache", [True, False])
def test_four_queries_five_branches_bounded_and_zero_writeback(tiny_qwen, use_cache):
    model = NativeQwenGlowPatchSpatialDecoderAuxModel(qwen=tiny_qwen, tokenizer=None,
        qwen_hidden_size=32, flow_dim=32, num_flow_steps=4, dropout=0., bev_grid_size=4,
        reference_schema="simple3", aux_weights={}, decoder_aux_weights={}, num_spatial_bias_experts=5,
        bias_gate_temperature=1., safe_bias_residual_init=.05, safe_bias_residual_max=.25,
        bias_norm_max_ratio=.15, spatial_query_count=4).eval()
    ids = torch.tensor([[2, 1, 1, 3, 4, 5, 6, 7, 8, 9]])
    grid = torch.tensor([[1, 2, 4]])
    pixels = torch.randn(8, 24)
    if use_cache:
        image = torch.randn(2, 32)
        deep = [torch.randn(2, 32), torch.randn(2, 32)]
    else:
        with torch.no_grad():
            vision = tiny_qwen.model.get_image_features(pixels, grid, return_dict=True)
        image = torch.cat(vision.pooler_output)
        deep = list(vision.deepstack_features)
    captures = []
    def capture(module, args, kwargs):
        captures.append({"image": kwargs["inputs_embeds"][ids == 1].detach().clone(),
                         "deep": [x.detach().clone() for x in kwargs["deepstack_visual_embeds"]],
                         "queries": kwargs["inputs_embeds"][0, 4:8].detach().clone()})
    handle = tiny_qwen.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=None if use_cache else pixels,
                  image_grid_thw=grid, mm_token_type_ids=(ids == 1).long(),
                  spatial_query_positions=torch.tensor([[4, 5, 6, 7]]),
                  cached_image_embeds=image if use_cache else None, cached_deepstack_features=deep if use_cache else None)
    with torch.no_grad():
        active = model._qwen_forward_with_spatial_bias(spatial_bias=torch.full((2, 32), 100.), **kwargs)
        zero = model._qwen_forward_with_spatial_bias(spatial_bias=torch.zeros(2, 32), **kwargs)
        flow = model.spatial_flow(descriptors=torch.randn(2, spatial.SPATIAL_DESCRIPTOR_DIM),
            token_sample_ids=torch.zeros(2, dtype=torch.long), reference_type_ids=torch.zeros(1, dtype=torch.long),
            query_pose_vec12=torch.zeros(1, 12), batch_size=1, pose_input_mask=torch.ones(1))
    handle.remove()
    delta = captures[0]["image"] - image
    assert torch.count_nonzero(delta) > 0
    assert (delta.norm(dim=-1) <= image.norm(dim=-1) * .15001).all()
    for actual, original in zip(captures[0]["deep"], deep):
        torch.testing.assert_close(actual - original, delta, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(captures[1]["image"], image, rtol=0, atol=0)
    for actual, original in zip(captures[1]["deep"], deep):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(captures[0]["queries"], model.spatial_query_role)
    assert active["spatial_query_hidden_states"].shape == (1, 4, 32)
    assert zero["safe_spatial_bias_abs_mean"].item() == 0.
    assert len(model.spatial_flow.flow_blocks) == 4
    assert flow["expert_biases"].shape == (2, 5, 32)
    assert flow["bias_gate_probs"].shape == (1, 5)
    metric = [flow["metric_" + name + "_pred"] for name in
              ["scene_scale", "room_extent", "camera_displacement", "object_size", "object_pair_distance", "camera_object_distance", "object_count"]]
    assert torch.cat(metric, dim=-1).shape == (1, 17)


def test_basic_noflow_reaches_both_visual_paths(tiny_qwen):
    model = basic.NativeQwenGlowPatchSpatialModel(qwen=tiny_qwen, tokenizer=None,
        qwen_hidden_size=32, flow_dim=32, num_flow_steps=4, dropout=0., bev_grid_size=4,
        reference_schema="simple3", aux_weights={}, disable_spatial_flow_bias=True).eval()
    ids = torch.tensor([[2, 1, 1, 3, 4]])
    image = torch.randn(2, 32)
    deep = [torch.randn(2, 32), torch.randn(2, 32)]
    captures = []
    def capture(module, args, kwargs):
        captures.append((kwargs["inputs_embeds"][ids == 1].detach(), kwargs["deepstack_visual_embeds"]))
    handle = tiny_qwen.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    raw = torch.ones(2, 32, requires_grad=True)
    disabled = model.spatial_bias_for_qwen(raw)
    assert not disabled.requires_grad and torch.count_nonzero(disabled) == 0
    with torch.no_grad():
        model._qwen_forward_with_spatial_bias(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=None,
            image_grid_thw=torch.tensor([[1, 2, 4]]), mm_token_type_ids=(ids == 1).long(),
            spatial_bias=disabled, cached_image_embeds=image, cached_deepstack_features=deep)
    handle.remove()
    torch.testing.assert_close(captures[0][0], image, rtol=0, atol=0)
    for actual, original in zip(captures[0][1], deep):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert len(model.spatial_flow.flow_blocks) == 4


def test_decoder_checkpoint_rejects_missing_and_unknown_readouts(tiny_qwen):
    model = NativeQwenGlowPatchSpatialDecoderAuxModel(qwen=tiny_qwen, tokenizer=None,
        qwen_hidden_size=32, flow_dim=32, num_flow_steps=4, dropout=0., bev_grid_size=4,
        reference_schema="simple3", aux_weights={}, decoder_aux_weights={}, num_spatial_bias_experts=5,
        bias_gate_temperature=1., safe_bias_residual_init=.05, safe_bias_residual_max=.25,
        bias_norm_max_ratio=.15, spatial_query_count=4)
    state = model.decoder_aux_state_dict()
    missing = dict(state)
    del missing["metric_consistency_query_head.0.weight"]
    with pytest.raises(KeyError, match="metric_consistency_query_head.0.weight"):
        model.load_decoder_aux_state_dict(missing)
    with pytest.raises(KeyError, match="unknown_readout"):
        model.load_decoder_aux_state_dict({**state, "unknown_readout": torch.ones(1)})


def test_grpo_checkpoint_restores_saved_sigreg_without_enabling_loss(tiny_qwen, tmp_path, monkeypatch):
    spatial.apply_lora_to_qwen_linears(tiny_qwen, target_modules={"q_proj"}, r=2, alpha=4., dropout=0.)
    models = []
    for weight in [.03, 0.]:
        models.append(NativeQwenGlowPatchSpatialDecoderAuxModel(qwen=tiny_qwen, tokenizer=None,
            qwen_hidden_size=32, flow_dim=32, num_flow_steps=4, dropout=0., bev_grid_size=4,
            reference_schema="simple3", aux_weights={}, decoder_aux_weights={}, num_spatial_bias_experts=5,
            bias_gate_temperature=1., safe_bias_residual_init=.05, safe_bias_residual_max=.25,
            bias_norm_max_ratio=.15, spatial_query_count=4,
            spatial_query_sigreg_loss_weight=weight, spatial_query_sigreg_projection_dim=8))
    source, target = models
    state = {key: value.clone() for key, value in source.decoder_aux_state_dict().items()}
    payload = {"format_version": decoder.FORMAT_VERSION,
        "args": {"reference_schema": "simple3", "spatial_query_count": 4,
                 "spatial_query_sigreg_projection_dim": 8},
        "spatial_flow_state_dict": source.spatial_flow.state_dict(),
        "decoder_aux_state_dict": state, "qwen_lora_state_dict": spatial.lora_state_dict(tiny_qwen)}
    path = tmp_path / "glow_patch_spatial.pt"
    path.write_text("synthetic checkpoint fixture")
    monkeypatch.setattr(decoder.torch, "load", lambda *args, **kwargs: payload)
    decoder.load_patch_spatial_checkpoint(path, target)
    assert target.spatial_query_sigreg_loss_weight == 0.
    assert not any(p.requires_grad for p in target.spatial_query_sigreg_projection.parameters())
    restored = target.decoder_aux_state_dict()
    assert set(restored) == set(state)
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_grpo_parser_supplies_shared_loader_contract():
    import ast
    import inspect
    from quasar import train_native_qwen3vl8b_decoderaux_numeric_grpo as grpo
    args = grpo.build_parser().parse_args(["--train-jsonl", "train.jsonl", "--val-jsonl", "val.jsonl",
        "--checkpoint-path", "checkpoint", "--output-dir", "output", "--qwen-model-path", "qwen"])
    # 检查两个真实入口之间的参数合同，防止到8B加载阶段才报属性缺失。
    tree = ast.parse(inspect.getsource(decoder.load_processor_and_qwen))
    fields = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name) and node.value.id == "args"}
    assert not fields - set(vars(args))
