# Public VSI input protocol v1

This protocol was fixed on 2026-10-08, before any locator inference or benchmark answer generation. Run selector: `public-vsi-v1`. It defines a new public evaluation and does not reconstruct the March Base run; the historical profiles, model weights and scorer are unchanged.

## Inputs and sampling

Use VSI `test.jsonl` revision `bdcadb3fea447621a828a24911801faba3587c12`, the package's ordered 5,130-member list and registered final 32 frames. All eight aggregate tasks remain. Samples are selected from question text, source and type only; `SAMPLES.json` freezes twelve IDs before inference. Six cover the nonsemantic task types; three cover directional easy/medium/hard across the three video sources; three cover ordinary robot facing, robot beginning by/facing, and robot back-to expressions. No sample is replaced after failure. `COVERAGE.tsv` enumerates all members and all 1,162 semantic expressions.

## Base

Official `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; original question, options and units, with no four-option expansion. Registered 32 RGB frames, 50,176 maximum pixels per image, bf16, sdpa, seed 0, greedy, 64 new tokens. The package's direct-native prompt is used with the Base question/options. Raw RGB goes through the official processor and `generate`; no CUT3R or reference is used. Reader verification uses processor inputs only; full answering and scoring use the commands in [PUBLIC_VSI.md](PUBLIC_VSI.md).

## Estimated QUASAR reference

Official inference rows contain no numeric question viewpoint. The upstream scene metadata includes object boxes and measurement/answer-related quantities, not a question-to-CUT3R transform. It is excluded from the estimator. No private object cache, relation truth, options, target question clause or answer enters localization.

One frozen route uses the same pinned official Qwen as a 2D object locator, without trained project weights. It sees the 32 selected RGB images after the pinned CUT3R resize/crop, and only each question's initial station/facing clause. It returns unique/ambiguous/missing status, selected 0-based frame and normalized [0,1000] bounding box for each anchor. Locator settings: bf16/sdpa, seed 0, greedy, 1,024 tokens, 262,144 maximum pixels per image. Malformed, ambiguous or missing outputs are retained as failed input records; no retry with another prompt/model or answer-dependent selection.

Geometry is regenerated from the complete registered CUT3R sequence (512 input, pinned code/weight), then uses the original selected 32 frames. Locator crop pixels are exactly those of CUT3R's public `load_images`; bbox centers select the central half of the box on its patch grid. Require at least three valid finite patches; use their componentwise median as the observed object-surface anchor. This is an estimated visible-surface position, not a ground-truth object center or human foot position.

For facing A from B, translation is B's estimated anchor; forward is A−B projected perpendicular to the estimated up vector. Up is the normalized negative mean of the selected camera y axes (upright handheld-camera assumption, not recovered gravity). Require camera-up concentration ≥0.8 and nondegenerate projected direction. Coordinate columns are right = forward × up, down = forward × right, forward. Units remain CUT3R reconstruction units; no metric or ScanNet-world claim is made. The rigid pose maps query coordinates into this exact reconstruction; no cross-scene transform is imported.

For back-to-object wording, a position alone does not identify the object's front direction. The same locator must additionally return a rectangle on the visible front face of that object. Fit a plane to finite patches inside its central half; require at least six patches, second singular value >1e−5 and smallest/second ratio ≤0.2. Orient the plane normal toward the observing camera, then project onto the estimated horizontal plane. Translation remains the station surface anchor. If the front face cannot be identified or the plane check fails, that sample is blocked. No substitute facing object or axis is used.

Every record preserves parsed object text, raw locator output, frames/crops, patch indices, point values, up estimate, plane fit when applicable and final matrix. Independent review checks visible object identities, front face/heading, and recomputes pose arithmetic. A valid matrix alone does not certify object identity. All semantic sample failures stay in the report and prevent declaring the two QUASAR routes ready.

## Acceptance and scope

Acceptance runs real-video extraction and CUT3R on the fixed samples and checks model readers, original Base options, answer isolation, coordinate gauge/rigid-matrix rejection, missing evidence and resume. CPU tests use synthetic fixtures and are reported separately from real evidence. Only verified routes are marked ready for a full run; full 1,162-object localization remains unverified even if the samples pass.
