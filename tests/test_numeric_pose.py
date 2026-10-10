"""检出全题开关漏类型、改坏原样本、混合批或投影偏置仍带参照的问题。"""
from dataclasses import dataclass
import unittest
import torch
from quasar.eval_native_qwen3vl2b_glow_patch_spatial_decoder_aux_generate import Gate1PoseInterventionDataset, Gate1GenerationCollator
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import QueryReferenceSpatialLatentFlow, build_patch_descriptor, audit_visual_cache_rows
from pathlib import Path

@dataclass(frozen=True)
class Sample:
    sample_id: str
    reference_type_id: int
    effective_pose_input_mask: bool
    query_pose: torch.Tensor

class FullPoseOffTest(unittest.TestCase):
    def test_all_types_invalid_and_repeat_restore(self):
        base=[Sample(str(i),i%3,bool(i%2),torch.eye(4)) for i in range(6)]
        for _ in range(3):
            current=Gate1PoseInterventionDataset(base,arm="current",reference_family="all",permutation_by_sample_id={})
            off=Gate1PoseInterventionDataset(base,arm="hard_off",reference_family="all",permutation_by_sample_id={})
            for i,original in enumerate(base):
                self.assertIs(current[i],original)
                self.assertFalse(off[i].effective_pose_input_mask)
                self.assertEqual(off[i].reference_type_id,original.reference_type_id)
                self.assertTrue(torch.equal(off[i].query_pose,original.query_pose))
                self.assertEqual(original.effective_pose_input_mask,bool(i%2))

    def test_descriptor_keeps_observation_and_zeros_reference(self):
        kw=dict(points=torch.arange(12).float().reshape(2,2,3),valid=torch.ones(2,2).bool(),
                confidence=torch.ones(2,2),source_pose=torch.eye(4),scene_center=torch.zeros(3),scene_scale=torch.tensor(2.))
        p=torch.eye(4); q=torch.eye(4);q[:3,3]=torch.tensor([9.,8.,7.])
        a=build_patch_descriptor(**kw,query_pose=p,pose_input_mask=False)
        b=build_patch_descriptor(**kw,query_pose=q,pose_input_mask=False)
        current=build_patch_descriptor(**kw,query_pose=q,pose_input_mask=True)
        self.assertTrue(torch.equal(a,b));self.assertEqual(torch.count_nonzero(a[...,6:12]).item(),0)
        self.assertTrue(torch.equal(a[...,:6],current[...,:6]));self.assertTrue(torch.equal(a[...,12:],current[...,12:]))

    def test_projection_bias_mixed_batch(self):
        torch.manual_seed(0)
        model=QueryReferenceSpatialLatentFlow(descriptor_dim=28,qwen_hidden_size=16,flow_dim=16,
            num_flow_steps=4,dropout=0.,bev_grid_size=4,reference_type_count=3,reference_schema="simple3",
            num_spatial_bias_experts=5,bias_gate_temperature=1.,safe_bias_residual_init=.05,
            safe_bias_residual_max=.25,bias_norm_max_ratio=.15).eval()
        desc=torch.randn(6,28);desc[:,6:12]=0
        kw=dict(descriptors=desc,token_sample_ids=torch.tensor([0,0,1,1,2,2]),reference_type_ids=torch.tensor([0,1,2]),batch_size=3,pose_input_mask=torch.tensor([0.,1.,0.]))
        a=torch.randn(3,12);b=a.clone();b[[0,2]]=15
        x=model(**kw,query_pose_vec12=a)["spatial_bias"]
        y=model(**kw,query_pose_vec12=b)["spatial_bias"]
        self.assertTrue(torch.equal(x,y))
        kw["pose_input_mask"]=torch.zeros(3)
        model(**kw,query_pose_vec12=b)
        self.assertEqual(model.poseoff_projection_max_abs,0.)

    def test_missing_cache_refused(self):
        with self.assertRaises(FileNotFoundError):
            audit_visual_cache_rows(rows=[{"sample_id":"missing","scene":{},"inputs":{},"refs":{"pose":{},"geometry":{"cut3r_npz_path":"/absent.npz"}}}],
                visual_cache_root=Path(__file__).parent/"absent_cache",scannet_frame_root=None,
                max_visual_frames=32,qwen_model_path=Path("/qwen"),max_pixels=50176,split_name="test")

if __name__ == "__main__":
    unittest.main()
