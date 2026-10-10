# Construct evaluation inputs

Evaluation starts from the chosen benchmark's official questions and videos, plus the official backbone and selected added weights. Training annotations are not required. Use the fixed versions and layouts in [main-table downloads](../data/TABLE12_DOWNLOADS.md), install the [checked environment](ENVIRONMENT.md), and edit `configs/reviewer_paths.example.json` into `reviewer_paths.json`. Paths resolve relative to that JSON file, independently of the shell's working directory.

## Scene and camera references

| Route | Frame sequence and reference | Coordinates and units |
| --- | --- | --- |
| Public Base VSI | Registered video extraction, then 32 model frames; original question/options | RGB only; no numerical reference |
| VSI scene reference | CUT3R reconstructs the complete registered candidate sequence; the model selects its 32 registered frames; reference origin is the mean of the selected observation's valid points | One CUT3R reconstruction per video; fixed observation axes; arbitrary reconstruction units, not assumed metres |
| VSTI camera reference | The registered 32-frame sequence; question frame index selects the corresponding CUT3R camera-to-world matrix | Same reconstruction as observed points; camera local axes are right/down/forward; arbitrary reconstruction units |
| Native ScanNet/SQA3D | Original measured depth and camera poses; original frame IDs and finite-pose selection | ScanNet world; depth millimetres divided by 1000 for points; original situation position and yaw follow the recorded SQA3D conversion |

The exact candidate/model IDs are in `data/manifests/table12_{vsi,vsti}_frames.jsonl`. Array row, `sample_order` and `embedding_row_index` must agree. A question's camera index is zero-based within the final selected sequence; it is not a timestamp or an index into all decoded video frames. The source loader uses its retained question parser. Its known limitations are listed in [release scope](RELEASE_SCOPE.md#known-limitations).

For a ready public route, the following runs preparation, answering and independent scoring. `--smoke` uses the supplied fixed sample order and does not produce a main-table result.

```bash
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role quasar_grpo --smoke --stage prepare
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role quasar_grpo --smoke
quasar-reviewer --paths reviewer_paths.json --dataset vsi --role base --protocol public-vsi-v1 --smoke
```

For a single scene-reference example, [generation](../data/GENERATION.md#original-questions-and-frames) supplies the official VSI video, exact 32-frame plan and commands. Native depth/camera and SQA3D examples are separate sections on that page. They do not redefine the main-table frame manifest.

The main-table spatial layout under `run_root/smoke/vsti/prepared/spatial/` is:

```text
qa.jsonl, plan.jsonl          scoring QA and source-row/frame bindings
geometry_frames.jsonl       ordered original frame identities and RGB paths
geometry/video_npz/*.npz    locally reconstructed points, masks and camera poses
prepared/inference.jsonl    question, options, RGB paths, geometry and reference
prepared/labels.jsonl       answers for separate scoring
features/*.pt              locally computed Qwen observation features
```

`source_row` identifies an occurrence in the ordered QA; repeated training IDs are not merged. Inference `answer` contains only `type`, `options` and optional `numeric_unit`. `refs.geometry` locates the new observation; `refs.simple3_aux` gives the reference type and matrix. Model input excludes gold answers, object supervision and training targets. Scene normalization uses the existing valid-point mean and maximum coordinate span with minimum scale 1; that minimum does not establish a metric unit.

## A supplied station and heading

`quasar-construct-reference` accepts a known station anchor, an explicit station offset, a facing point, an up direction and a sourced coordinate transform. It does not identify objects in video. The offset specifies where the observer stands relative to the station object; an object centre alone is not an observer position. Use only reference information available under the intended inference protocol. Test answers, answer-derived relations and evaluation-only object truth cannot establish that protocol.

The JSON schema is shown by `data/examples/reference_definition.synthetic.json`. Required fields are:

- Identity: `source_row`, `sample_id`, `video_id`, full ordered `frame_ids`.
- Provenance: `source`, `kind`, `source_gauge`, `geometry_gauge`, `source_units`, `geometry_units`.
- Vectors: `station_point`, `station_offset`, `facing_point`, `up`.
- Alignment: `source_to_geometry` with positive `scale`, proper 3×3 `rotation`, three-coordinate `translation` and `source` describing how alignment was obtained. Identity alignment must also be explicit.

The point conversion is $p_g=sR(p_s+o_s)+t$. Directions are rotated without translation. Forward is the facing-point direction projected onto the plane perpendicular to up. The output matrix has right, down and forward columns and transforms reference-local points into the observation coordinates. The transform is supplied, not estimated by this tool; copying a coordinate-system name does not align two reconstructions.

```bash
quasar-construct-reference --definition runs/input/station.json \
  --frame-manifest runs/geometry_frames.jsonl \
  --geometry-npz runs/geometry/video_npz/VIDEO_ID.npz \
  --output-jsonl runs/input/reference.jsonl
quasar-prepare-public-inputs --qa-jsonl runs/input/qa.jsonl \
  --input-plan runs/input/plan.jsonl --frame-manifest runs/geometry_frames.jsonl \
  --geometry-root runs/geometry --reference-jsonl runs/input/reference.jsonl \
  --output-root runs/prepared --max-visual-frames 32
```

The constructor verifies video, ordered frame IDs, candidate IDs and image-path binding against the same geometry file. It rejects missing vectors/provenance, answer fields, nonfinite values, invalid rotations and degenerate headings. `reference.jsonl` contains only the preparer's supported reference fields; `reference.audit.json` retains the transform and calculation. Source permissions and object identity require evidence from the supplied material, beyond matrix validation.

This standalone teaching example needs no downloaded data or GPU:

```bash
python data/examples/create_reference_fixture.py --output-root runs/reference_teaching
quasar-construct-reference --definition runs/reference_teaching/definition.json \
  --frame-manifest runs/reference_teaching/frames.jsonl \
  --geometry-npz runs/reference_teaching/binding.npz \
  --output-jsonl runs/reference_teaching/reference.jsonl
```

Its coordinates and image are synthetic, labelled `synthetic_teaching`; it checks the arithmetic only and supplies none of the 1,162 missing VSI viewpoints ([public VSI checks](PUBLIC_VSI_VALIDATION.md)).

## Load, answer and score

The [checkpoint guide](../checkpoints/README.md) supplies extraction and role-specific loading commands. The main-table reviewer runs these steps automatically. For a standalone prepared input, export features with the commands in [generation](../data/GENERATION.md#features-and-answering), then bind the matching recorded role to your paths. Keep labels outside the generation file. Use `quasar-score-retained --eval-jsonl` with the independently held native scoring rows; `prepared/labels.jsonl` alone omits task metadata, so use the reviewer's full `spatial/labels.jsonl` for this command.

The explicit [numeric-reference comparison](NUMERIC_POSE.md) is separate from the main-table entry.
