# BindGeorge

**Protein binder prioritization across multiple representations.**

Which protein representations help distinguish promising binders once they
have passed a structure-confidence filter? BindGeorge explores this question
through a matched benchmark of sequence embeddings, AlphaFold internal states,
interface geometry, molecular surfaces and physical descriptors.

The repository brings together compact readout models, training and evaluation
scripts, aggregate results, and synthetic examples for exploring the code.

## What's inside

- **Sequence and structure representations:** ESMC embeddings and AlphaFold 2/3
  single and pair states, including structure-conditioned AF3 refined states.
- **Geometry, surfaces and physics:** interface descriptors, an equivariant atom
  model, dMaSIF surface readouts and Rosetta-derived features.
- **A shared evaluation setup:** grouped out-of-fold evaluation for learned
  readouts, alongside scalar confidence baselines.

The primary benchmark compares 18 entries on 2,881 examples, including 756
positives, after an AF3 iPAE_min <= 2.5 angstroms filter. The full comparison is
in [the results table](results/main_results.tsv), with evaluation details in
[the protocol](docs/PROTOCOL.md).

These results describe binder prioritization for known targets. Experimental
labels come from heterogeneous campaigns; they are not uniform affinity
measurements. The comparison does not establish prospective hit rates or a
single best representation.

## Get started

From the repository root, view the results and check the release contents with
Python 3.11 or newer. These two commands use only the standard library:

```bash
python -m scripts.show_results
python -m scripts.audit_release
```

To run the synthetic CPU tests, install a platform-compatible PyTorch build in
a dedicated environment, then install the core dependencies:

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

The tests use small artificial inputs and require no dataset, pretrained
weights or GPU. See [environment notes](docs/ENVIRONMENT.md) for dependencies
and settings for shared machines. Graph models additionally require e3nn.

## Explore the code

| Path | Contents |
|---|---|
| [`training/`](training/) | Representation loaders and readout models |
| [`scripts/`](scripts/) | Training, evaluation and analysis entry points |
| [`tests/`](tests/) | Synthetic cache, aggregation and model checks |
| [`results/main_results.tsv`](results/main_results.tsv) | Aggregate results for the 18-entry benchmark |
| [`docs/`](docs/) | Evaluation protocol, input formats and release notes |

For the research entry points, start with their available options:

```bash
python -m scripts.train_multimodal_v2 --help
python -m scripts.train_representation_benchmark_completion_v1 --help
```

Training requires separately prepared data, fold memberships and extracted
representations. The original datasets, feature caches and trained checkpoints
are not included. Some orchestration scripts also depend on files from the
original research workspace; the repository is not yet a self-contained
training pipeline. See [input requirements](docs/INPUTS.md) and
[release notes](docs/RELEASE_STATUS.md) before running those scripts.

Upstream tools and model weights are obtained separately under their respective
terms; links are collected in [third-party software](docs/THIRD_PARTY.md).

---

[Fred (Yanbo) Jing](https://github.com/Fred-Jing) · Upcoming poster presentation at NIPS 2026.
