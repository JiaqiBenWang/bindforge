# BindForge

One pipeline for **protein binder design → complex structure prediction → MD stability validation**, wrapped so *everyone can use it* — no local GPU required.

BindForge chains three stages:

1. **Design** — generate candidate binder sequences for a target protein (de novo).
2. **Predict** — fold the target + binder complex with an open-source AlphaFold3 alternative (**[Boltz-2](https://github.com/boltz-community/boltz)** / **[Chai-1](https://github.com/chaidiscovery/chai-lab)**), avoiding Google's weight-license barrier.
3. **Validate** — run local **[OpenMM](https://openmm.org)** MD (implicit GBSA by default) to measure whether the binder stays in its predicted pose, retains interface contacts, and binds favorably, then rank candidates.

The heavy compute runs through **online APIs** (NVIDIA NIM hosted Boltz-2, boltz.bio), so it works on a plain laptop — including Windows with no GPU/CUDA/Docker.

---

## Install

```bash
git clone https://github.com/JiaqiBenWang/bindforge.git
cd bindforge
pip install -e .            # core
pip install -e ".[md]"      # + OpenMM (MD validation)
pip install -e ".[web]"     # + FastAPI/uvicorn (web UI)
```

Requires Python ≥ 3.8.

---

## Quick start (no API keys)

The `--dry-run` flag uses deterministic **mock** providers, so the whole pipeline
runs end-to-end with zero configuration (real OpenMM MD included):

```bash
bindforge run --target tests/fixtures/target.pdb \
    --n-designs 4 --length 50-60 --md-top 2 --md-ns 0.2 --dry-run
```

This writes a ranked table to `results/ranking.csv` / `results/ranking.json`
plus each complex PDB and MD snapshot.

You can pass a **raw sequence** instead of a file too:

```bash
bindforge run --target "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP" \
    --n-designs 4 --length 50-60 --md-top 2 --md-ns 0.2 --dry-run
```

---

## Real runs (online APIs)

Copy `.env.example` to `.env` and add your keys:

- **NVIDIA NIM (Boltz-2 prediction)** — get a key at <https://build.nvidia.com/mit/boltz2>, set `NVIDIA_API_KEY`.
- **boltz.bio (BoltzGen de novo design + prediction)** — <https://boltz.bio>, set `BOLTZ_BIO_API_KEY`.

Then run without `--dry-run`, picking providers explicitly:

```bash
bindforge run --target my_target.pdb \
    --design-provider boltzgen --structure-provider nvidia_boltz2 \
    --n-designs 8 --length 50-80 --md-top 5 --md-ns 5.0
```

> Chai-1 has no official programmatic API yet; a `chai` provider adapter is
> reserved as a fallback (see `binderforge/providers/chai.py`).

---

## CLI

```bash
bindforge run      # full pipeline: design -> predict -> MD -> rank
bindforge design   # design binder sequences only (writes FASTA)
bindforge predict  # predict a single target+binder complex
bindforge md       # MD-validate an existing complex PDB/CIF
bindforge serve    # start the web UI
```

Key `run` flags: `--target`, `--n-designs`, `--length 50-80`, `--hotspot`,
`--design-provider`, `--structure-provider`, `--md-top`, `--md-ns`, `--dry-run`,
`--results-dir`, `--seed`.

---

## Web UI

```bash
pip install -e ".[web]"
bindforge serve            # http://127.0.0.1:8000
```

The web UI requires **login** — register with an email + password (accounts and
sessions are stored locally in `data/`; passwords are PBKDF2-hashed and sessions
are HMAC-signed, no external auth dependency). Then **upload a target PDB / CIF
/ FASTA** (or paste a raw sequence), set the parameters, and hit *Run pipeline*.
The UI shows live MD progress, a job list scoped to your account, and the ranked
results table with per-component scores.

**Scope & limits (shown in the UI):** Boltz-2 supports up to ~4096 residues/chain
(≈2000 total on a single GPU); Chai-1 up to 2048 tokens locally / 1024 residues
via API. MD is CPU-bound, so keep target + binder under ~500 residues.
Post-translational modifications (glycosylation, phosphorylation, …) are **not
modeled in the MD stage** — non-standard residues are stripped before
simulation; only the 20 standard amino acids (plus MSE/SEP/TPO) are kept.

The backend runs the same `binderforge.pipeline` code in a background thread; a
3D (NGL/Mol*) structure viewer is the planned upgrade path.

---

## Scoring

Candidates are ranked by a weighted composite:

```
score = 0.35·confidence + 0.25·stability + 0.25·binding + 0.15·pose
```

- **confidence** — mean ipTM/pTM (fallback pLDDT) from the predictor.
- **stability** — MD interface contact retention (fraction of initial contacts kept).
- **binding** — `-ΔG / 40` from MM-GBSA (kJ/mol).
- **pose** — `exp(-RMSD_mean / 0.5)` (did the binder drift from its predicted pose?).

Component scores are `null` (not zero) when the underlying quantity was not
measured — a failed MD run, or a starting pose with no interface to track. The
composite is renormalised over the measured components, and MD-validated
candidates are always ranked ahead of unvalidated ones.

MD metrics per candidate: binder RMSD (vs. predicted pose, superposed on the
target each frame), heavy-atom interface contact retention, MM-GBSA ΔG estimate,
and per-residue RMSF. See `binderforge/md/`.

---

## Architecture

```
binderforge/
  cli.py             # argparse CLI (run/design/predict/md/serve)
  pipeline.py        # orchestration
  schemas.py         # dataclasses (Binder / ComplexPrediction / MDResult)
  config.py          # env/.env config
  io.py              # FASTA/PDB parsing + full-atom peptide builder
  scoring.py         # composite scoring & ranking
  providers/         # design + structure provider adapters (mock / nvidia_nim / boltzbio / chai)
  md/                # OpenMM engine (implicit GBSA default; explicit stub) + metrics
server/              # FastAPI backend + static frontend
tests/               # pytest
```

- **Implicit GBSA MD** (default): Amber ff14SB + OBC2, `minimize → NVT equilibrate → production` on CPU; `--platform CUDA` if you have a GPU.
- **Explicit TIP3P** is reserved for phase 2 (`binderforge/md/explicit.py`).

---

## License

MIT.
