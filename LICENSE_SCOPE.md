# License scope

Copyright in original QUASAR contributions remains with their respective rights holders. The project is attributed as **QUASAR contributors** during anonymous review; this is an attribution label, not a transfer of ownership or a claim about third-party material.

| Material | Applicable terms |
| --- | --- |
| Original QUASAR code, accompanying project documentation and build configuration | Apache-2.0; full text in [LICENSE](LICENSE) |
| Qwen/Transformers code adapted in the visual-forward paths | Apache-2.0 with the original notices and the modifications identified in [NOTICE](NOTICE) and the affected source files |
| QUASAR added model parameters in W01, W02, W05 and W06 | CC-BY-NC-4.0; full text in [LICENSE-WEIGHTS](LICENSE-WEIGHTS), with the scope below and inside each archive |
| Included questions, answers and source-derived data records | Each source's own terms, identified in [data/TERMS.md](data/TERMS.md) and the per-record license fields; neither project license replaces them |
| External libraries, the Qwen backbone/tokenizer, CUT3R and original media | Their providers' terms; see [third-party notices](third_party/README.md). These components are acquired separately |

## Added weights

CC-BY-NC-4.0 applies only to the project's added model parameters and their accompanying project description in the four first-release archives. It permits noncommercial use, sharing and adaptation under the license, with attribution and an indication of changes. Noncommercial use is not restricted to academic research. Commercial use of the project parameters requires separate authorization. No paper-citation condition or additional research-only restriction is imposed.

For attribution, retain the project name, model ID, license notice and the source link, and credit **QUASAR contributors** (see the attribution label above). The label asserts no personal identity, institution or endorsement.

The model is the separately acquired official Qwen3-VL-8B-Instruct backbone plus the project's added parameters. The backbone retains Apache-2.0 and is not relicensed as CC-BY-NC-4.0. CUT3R retains CC-BY-NC-SA-4.0, including its share-alike condition for covered adaptations. Data licenses and provider access conditions remain attached to their own material. A project license grants only rights the project contributors can grant.

## Distribution metadata

The Python distribution's `License` metadata contains this scope statement rather than declaring that every file has one license. The source distribution includes QA with multiple source licenses; the wheel contains the Python implementation and license/notice documents, without QA or model parameters. The presence of a license text in the wheel does not apply that license to every wheel member. Keep the matching notice and scope documents with redistributions.
