# MACE-Based Equivariant Diffusion for Chirality-Aware Conformer Generation

A MACE (equivariant GNN) diffusion model that generates 3D conformers of
nonane, conditioned on a chirality/symmetry-measure descriptor. Built on
[ACEsuit/mace](https://github.com/ACEsuit/mace) and
[e3nn](https://github.com/e3nn/e3nn), following the EDM (Karras et al.)
preconditioning and Heun sampling scheme.

## Pipeline

```
data_pipeline/01_generate_conformers.py   nonane.sdf --dihedral scan + MMFF--> conf_i_j.sdf
data_pipeline/02_extract_chirality_labels.py   conf_*.sdf --RDKit + cosymlib--> chirality_data.csv
                                                                        │
                                                                        ▼
scripts/run_training.py    chirality_data.csv -> dataset -> model -> train -> checkpoints/*.pt
scripts/run_sampling.py    checkpoints/*.pt -> conditioned generation -> outputs/*.xyz
```

## Repo structure

```
├── data_pipeline/
│   ├── 01_generate_conformers.py       # RDKit dihedral-scan + MMFF relaxation
│   └── 02_extract_chirality_labels.py  # RDKit chiral centers + cosymlib symmetry measures
├── configs/
│   └── config.py                       # fixed hyperparameters, cutoffs, element tables
├── src/mace_diffusion/
│   ├── env_setup.py                    # determinism flags — import first, always
│   ├── dataset.py                      # .mol2 parsing, MaceMolecularDataset, angle/dihedral indices
│   ├── graph_utils.py                  # radius_graph_manual (diagnostic only, not used in forward())
│   ├── model.py                        # TimeEmbedding, MaceVectorReadout, MaceDiffusionWrapper
│   ├── diffusion.py                    # VE-SDE forward noising, EDM scalings, Karras sigma schedule
│   ├── train.py                        # F-space EDM loss, avg_num_neighbors, training loop
│   ├── sample.py                       # Heun ODE sampler, .xyz export
│   └── checkpoint.py                   # save/load — checkpoint is self-contained (full config included)
├── scripts/
│   ├── run_training.py                 # end-to-end training entry point
│   └── run_sampling.py                 # end-to-end sampling entry point
├── notebooks/                          # exploratory notebooks (not part of the package)
├── checkpoints/                        # saved model checkpoints (gitignored)
└── outputs/                            # generated .xyz structures (gitignored)
```

## Key design points worth knowing before reading the code

- **Two separate edge lists, two separate purposes.** `edge_index` (geometric,
  distance-cutoff-based, changes with noise level) is used for message
  passing. `bond_edge_index` (chemical, from the file's explicit bond list,
  fixed regardless of geometry) is used only for the angle/dihedral auxiliary
  quantities.
- **The candidate edge list is built once, with a deliberately oversized
  cutoff** (`LARGE_CUTOFF=15.0` vs. the real `R_MAX=8.0`), and only *filtered*
  (never rebuilt) at every noise level inside `forward()`. This assumes noise
  never displaces an atom enough to leave the candidate set — verify this
  margin empirically for your own `sigma_max` before trusting it.
- **Two different sigma schedules, on purpose.** The training schedule
  (log-uniform, `sigma_data`/`sigma_min`/`sigma_max` all derived from the
  actual training set's coordinate scale) needs broad, cheap coverage. The
  Karras sampling schedule needs to make the most of a small, expensive, fixed
  step budget — hence the different (rho-power) spacing.
- **Timestep conditioning uses addition; chirality conditioning uses
  concatenation + a learned projection.** The former is simpler and standard
  in diffusion literature; the latter is strictly more expressive, since
  concatenation lets every input dimension influence every output dimension
  independently rather than forcing a rigid one-to-one shift.
- **This is a conformer generator, not a de novo generator.** Atom count,
  atom identity, and the candidate topology are all taken from a template
  structure at generation time — only 3D positions are diffused. Generating
  atom count / identity / topology from scratch is planned future work.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python data_pipeline/01_generate_conformers.py
python data_pipeline/02_extract_chirality_labels.py
python scripts/run_training.py
python scripts/run_sampling.py
```
