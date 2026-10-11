# Input contracts and omitted artifacts

`B` is batch size; `Lt` and `Lb` are target and binder lengths. Padding masks
are boolean, True for valid positions. Targets may contain multiple chains;
retain chain boundaries and residue ordering in any feature producer.

| Input | Batched shape | Meaning |
|---|---|---|
| target/binder ESMC | `[B,Lt,2560]`, `[B,Lb,2560]` | Frozen ESMC-6B embeddings |
| target/binder single | `[B,Lt,384]`, `[B,Lb,384]` | AF3 mature/refined single, or compatible AF2 adapter input |
| target_binder_pair | `[B,Lt,Lb,128]` | Target-to-binder pair block |
| binder_target_pair | `[B,Lt,Lb,128]` | Opposite block already transposed into target/binder order |
| target/binder mask | `[B,Lt]`, `[B,Lb]` | Valid residue positions |
| geometry g4 | `[B,249]` | Ordered geometry/identity features, training-fold standardized |
| raw Rosetta | `[B,39]` | Ordered physical features with train-fitted imputation |
| dMaSIF points | variable points per partner, 82 channels | Partner-aware descriptor and geometry input |

Do not assume equal shape means equivalent model state. Mature and refined
activations have different extraction points. The benchmark's refined features
come after AF3 Confidence PairFormer and before scalar projections, for the
selected structure sample. No runtime extraction patch is included here.

For dMaSIF, each point contains own embedding1 (16), own embedding2 (16),
partner embedding2 (16), own1 times partner2 (16), absolute difference (16),
nearest cross-surface distance divided by 3 angstroms (1), and the own/nearest
partner surface-normal dot product (1). The primary K8 partner average uses
normalized Gaussian distance weights with width 3 angstroms. The input cache
builders and upstream checkpoint are not supplied in this first snapshot.

## Data and caches

Historical loaders specify file contracts in `training/data.py`,
`training/multimodal_data.py` and `training/interface_atom_data.py`. Sequence
caches use `metadata.json`, `sequence_manifest.tsv` and `embeddings.f16.npy`.
Real datasets, sequence manifests and tensors are deliberately absent. Test
fixtures create tiny artificial equivalents under temporary directories.

Research scripts retain assertions on the original 2,881-example cohort and
file layout beneath the package root. A missing-file error without those
inputs is expected; do not bypass the assertions to claim reproduction.
Use explicit CLI input/output options where offered. The serial worker uses
the current Python executable; set `BINDER_GRAPH_PYTHON` for a separate e3nn
environment. No sibling research directory is used as a fallback.

The preparation script also hashes original protocol and scheduler files that
are not included in this release: `experiments/representation_benchmark_completion_v1.md`,
`scripts/submit_representation_benchmark_completion_v1.py`, and the two
`slurm/run_representation_benchmark_completion*_v1.sbatch` files. Its original
fold manifest and reference run artifacts are also required. Supplying feature
caches alone is therefore insufficient to run the preparation and worker
workflow unchanged. Individual training entry points and the synthetic tests
have different requirements; see their CLI options and the environment notes.

## Weights and preprocessing

No checkpoints are included. Neural model output is a logit, not a validated
prospective probability. Any later weights package must carry exact adapter
configuration, channel order, mask/chain convention, fitting scope and hashes.
Geometry scalers and Rosetta imputation/estimator state are essential, not
optional metadata. Load only trusted checkpoints; historical pickle/joblib
loading code is not a security boundary. The smoke tests round-trip freshly
created tensor-only state dictionaries with `weights_only=True`.
