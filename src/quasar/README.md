# Implementation

The package contains 44 modules. Eighteen of them form the original dependency chain (see [SOURCE_VERSIONS.md](../../docs/SOURCE_VERSIONS.md)); the rest add data preparation, reference construction, the reviewer entry and retained-answer scoring. Existing function and class logic is preserved, with package imports and standalone-path setup adapted. Legacy `2b` filenames also support the recorded 8B configuration; model size follows the supplied backbone. Some module and checkpoint file names keep the development prefix `glow` (for example `glow_patch_spatial.pt`); they belong to the same QUASAR implementation.

The implementation retains:

- Read video/frame inputs, visual features, geometry, explicit references, and training-only targets with distinct validity rules.
- Construct aligned 28-component geometry descriptors.
- Apply the original four spatial updates and five branches, including metric feedback and branch mixing.
- Apply learned scaling and the relative norm bound; write the same effective correction to native and aligned DeepStack features.
- Insert four learned spatial summary embeddings and provide local/global decoder readouts for SFT.
- Run SFT and LoRA-only GRPO with stage-specific objectives.
- Generate answers for full and explicitly named control conditions; preserve actual effective checkpoint settings.
- Score retained answers with each documented dataset/subset aggregation.

Source chronology and its relation to the June training runs are documented in [SOURCE_VERSIONS.md](../../docs/SOURCE_VERSIONS.md). The data/reference/target readers are the actual implementations and are exercised by the CPU tests.
