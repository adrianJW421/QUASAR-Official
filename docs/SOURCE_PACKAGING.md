# Source payload and generated assets

`PUBLIC_FILES.txt` is the exact source-file list. It retains code, configuration, licensed validation/scoring QA, member order, target-availability records, original predictions/scores, source licenses and tests. The large training QA file is downloaded separately ([data guide](../data/DOWNLOAD_GUIDE.md#included-records-and-additional-inputs)) and excluded from Git and source archives. Features, estimated points/poses, runtime caches, model weights, provider media and local source maps are absent.

```bash
python scripts/package_source.py --output ./runs/release/quasar-source.tar.gz
```

The script uses only the explicit list, rejects generated directories and model/geometry/feature files, and checks that `MANIFEST.in` includes exactly the same files (including the README figures) for Python source distributions. Wheels contain Python code and the applicable license/scope/notice documents; obtain the source payload for data and recorded configurations. Distribution metadata retains the scope statement so that source QA is not described as wholly Apache-2.0. Tests write fixtures to the selected temporary directory; test results and runtime-validation receipts are not packaged.

Use `upstream/` for provider downloads and `runs/` for extraction, geometry, features, targets and answers. These and the named generated-data directories are ignored. Keep gold labels separate from answering inputs.

The released weight archives (W01, W02, W05 and W06) keep the model components in their original precision with the role and configuration metadata needed for loading. Optimizer/RNG state and computed features/geometry are excluded. Official backbones, frontends and tokenizers are downloaded from their providers. Original code uses Apache-2.0 and the project-added weights use CC-BY-NC-4.0.

Presentation images, such as the README figures, are not model point maps or features. Provider media, scan annotation and provider questions remain subject to their original terms.

[GENERATION.md](../data/GENERATION.md) lists the runnable input routes and the reference/target conversions that are missing.
