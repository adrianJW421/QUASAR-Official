"""用真实模型检查参照开关；这里不读答案，也不计算分数。"""
from dataclasses import replace
import json
import torch
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import move_batch_to_device


def assert_same_inputs(a, b, *, allow_pose=False):
    # 只比较生成真正读取的内容，训练监督张量不进入此次作答。
    keys = ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw",
            "spatial_query_positions", "cached_image_embeds", "cached_deepstack_features",
            "spatial_points", "spatial_token_valid", "spatial_token_sample_ids", "reference_type_ids")
    if not allow_pose:
        keys += ("query_pose_vec12", "effective_pose_input_mask", "spatial_descriptors")
    for key in keys:
        av, bv = a[key], b[key]
        if isinstance(av, (list, tuple)):
            assert len(av) == len(bv), key
            for x, y in zip(av, bv):
                assert torch.equal(x, y), key
        else:
            assert torch.equal(av, bv), key
    if allow_pose:
        for cols in (slice(0, 6), slice(12, None)):
            assert torch.equal(a["spatial_descriptors"][:, cols], b["spatial_descriptors"][:, cols])


@torch.no_grad()
def forward_evidence(model, batch):
    """直接走原空间模块和原Qwen前向，检查写回量与下一个字的完整概率前分数。"""
    flow = model.spatial_flow(
        descriptors=batch["spatial_descriptors"].float(),
        token_sample_ids=batch["spatial_token_sample_ids"].long(),
        reference_type_ids=batch["reference_type_ids"].long(),
        query_pose_vec12=batch["query_pose_vec12"].float(), batch_size=1,
        pose_input_mask=batch["effective_pose_input_mask"].float())
    bias = flow["spatial_bias"] * float(model.eval_spatial_bias_scale)
    output = model._qwen_forward_with_spatial_bias(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"), image_grid_thw=batch["image_grid_thw"],
        mm_token_type_ids=batch["mm_token_type_ids"], spatial_bias=bias,
        spatial_query_positions=batch["spatial_query_positions"].long(),
        cached_image_embeds=batch["cached_image_embeds"],
        cached_deepstack_features=batch["cached_deepstack_features"],
        global_3d_points=batch["spatial_points"], global_3d_valid=batch["spatial_token_valid"])
    logits = output["logits"][:, -1, :]
    return (bias.float().cpu(), logits.float().cpu(), bias.dtype, logits.dtype)


def verify_real_inputs(model, base_dataset, dataset, base_collator, collator, device, arm, output_dir):
    records = []
    # 用两个不同类别/状态的真实样本检查拼批，不把后一题的开关误用到前一题。
    mixed_indices=[0,len(dataset)-1]
    mixed=collator.collate_generation_prompts([dataset[i] for i in mixed_indices])
    assert mixed['reference_type_ids'].tolist()==[base_dataset[i].reference_type_id for i in mixed_indices]
    expected_masks=[0. if arm=='hard_off' else float(base_dataset[i].effective_pose_input_mask) for i in mixed_indices]
    assert mixed['effective_pose_input_mask'].tolist()==expected_masks
    (output_dir/'PREFLIGHT_BATCH.json').write_text(json.dumps({'passed':True,'ids':[dataset[i].sample_id for i in mixed_indices],
        'reference_type_ids':mixed['reference_type_ids'].tolist(),'effective_masks':expected_masks},indent=2)+'\n')
    del mixed
    for index in range(len(dataset)):
        original = base_dataset[index]
        sample = dataset[index]
        raw = base_collator.collate_generation_prompts([original])
        changed = collator.collate_generation_prompts([sample])
        assert_same_inputs(raw, changed, allow_pose=arm == "hard_off")
        if arm == "hard_off":
            # 连原始4x4站位一起替换，检查描述子和其他旁路没有藏着站位依赖。
            pose_a = torch.eye(4); pose_a[:3, 3] = torch.tensor([-3., 2., 1.])
            pose_b = torch.eye(4); pose_b[:3, :3] = torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
            pose_b[:3, 3] = torch.tensor([19., -11., 7.])
            for pose in (pose_a, pose_b):
                alternative = collator.collate_generation_prompts([replace(sample, query_pose=pose)])
                assert_same_inputs(changed, alternative)
        batch_a = move_batch_to_device(changed if arm == "hard_off" else raw, device)
        batch_b = move_batch_to_device(changed, device)
        if arm == "hard_off":
            batch_a = dict(batch_a, query_pose_vec12=torch.linspace(-3.,3.,12,device=device).view(1,12))
            batch_b = dict(batch_b, query_pose_vec12=torch.linspace(19.,-11.,12,device=device).view(1,12))
        a = forward_evidence(model, batch_a)
        repeat = forward_evidence(model, batch_a)
        b = forward_evidence(model, batch_b)
        item = {"sample_id": sample.sample_id, "arm": arm,
                "reference_type_id": sample.reference_type_id,
                "base_effective_mask": original.effective_pose_input_mask,
                "non_reference_inputs_equal": True, "raw_pose_replacement_equal": arm == "hard_off"}
        for j, name in enumerate(("spatial_bias", "next_token_logits")):
            delta = float((a[j]-b[j]).abs().max())
            noise = float((a[j]-repeat[j]).abs().max())
            # 沿用旧诊断的精度容差，并同时要求下一字的选择完全一致。
            scale = max(1.,float(a[j].abs().max()),float(b[j].abs().max()))
            tolerance = 2*noise + (8 if j == 0 else 2)*torch.finfo(a[j+2]).eps*scale
            item[name] = {"max_abs_diff":delta,"repeat_noise":noise,"tolerance":tolerance,"exact_equal":torch.equal(a[j],b[j])}
            assert delta <= tolerance, item
        assert torch.equal(a[1].argmax(-1),b[1].argmax(-1)), item
        item["next_token_equal"] = True
        if arm == "hard_off":
            assert model.spatial_flow.poseoff_projection_max_abs == 0
        item["passed"] = True
        records.append(item)
        (output_dir/"PREFLIGHT.json").write_text(json.dumps(records,indent=2)+"\n")
        print(json.dumps({"event":"preflight","sample_id":sample.sample_id,"passed":True}),flush=True)
