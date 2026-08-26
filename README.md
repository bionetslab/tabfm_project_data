# tabfm_project_data

Preprocessed data for the TABFM Bionets student project — *do tabular foundation models help on
small-cohort flow cytometry?*

Three public flow-cytometry benchmark datasets, reduced from **30.3 GB of raw FCS files (448 million
cells)** to **187 MB of ready-to-use arrays**. Everything needed for the project fits in a laptop's RAM;
no FCS parsing, no access to the source archive, and no GPU are required.

---

## Contents

| File | Cube shape | Subjects | Positives | Size |
|---|---|---|---|---|
| `aml_cells_n2000_seed0.npz` | (359, 8, 2000, 7) | 359 | 43 AML | 75 MB |
| `fc4_cells_n2000_seed0.npz` | (383, 2, 2000, 21) | 383 | 79 progressors | 96 MB |
| `heu_cells_n2000_seed0.npz` | (44, 7, 2000, 10) | 44 | 20 HEU | 16 MB |
| `manifest.json` | — | shapes and class balance for each cube | | 1 KB |

`seed0` names the random draw of cells used (see *How the data was preprocessed*, step 4). Only this one
draw is distributed here; a second, independent draw exists and can be provided on request if you want to
measure how much of your error bars come from the cell subsampling itself.

## Source datasets

All three are public on [FlowRepository](https://flowrepository.org) and were retrieved from a read-only
archival mirror of it (metadata snapshot 2026-07-28).

| Cube | Accession | Description | Raw |
|---|---|---|---|
| `aml` | [FR-FCM-ZZYA](https://flowrepository.org/id/FR-FCM-ZZYA) | FlowCAP-II AML challenge. 2,872 files = 359 subjects × 8 tubes, 7 parameters, ~30 k cells/file | 3.9 GB, 80.0 M cells |
| `fc4` | [FR-FCM-ZZ99](https://flowrepository.org/id/FR-FCM-ZZ99) | FlowCAP-IV HIV progression. 766 files = 383 subjects × (stimulated, unstimulated), 22 parameters | 20.4 GB, 232.0 M cells |
| `heu` | [FR-FCM-ZZZU](https://flowrepository.org/id/FR-FCM-ZZZU) | FlowCAP-II HEU vs UE. 308 files = 44 infants × 7 stimulation conditions, 11 parameters | 6.0 GB, 135.6 M cells |

Labels come from each dataset's own attachment (`AML.csv`, `MetaDataFull.csv`, `HEUvsUE.csv`), not from
any re-annotation by us.

## How the data was preprocessed

One streaming pass per dataset, ~15 s each. Nothing was written to the source archive and no FCS file was
ever extracted to disk — members were read straight out of the `.tar` at recorded byte offsets.

1. **Decode.** HEADER/TEXT/DATA are parsed and the DATA segment converted to **linear** `float32`. Three
   FCS traps are handled explicitly, and getting any of them wrong silently corrupts the values:
   - `$PnE` log amplification — `"f1,f2"` with `f1 > 0` means the stored integer is a *log* value, and
     `f2 == 0` must be read as `f2 == 1` per the FCS3.1 spec. **The AML dataset is FCS2.0 with log
     amplification and depends on this.**
   - `$PnR` range — integer data is often stored in wider words than it uses (`$PnB = 16` with
     `$PnR = 1024` means 10 valid bits); the high bits are undefined and are masked off.
   - `$PnB` bit width — usually a uniform 16 or 32, occasionally mixed per channel.
2. **Drop the `Time` channel.** It is an acquisition artefact, not a measurement. This is why `fc4` has 21
   channels rather than 22 and `heu` has 10 rather than 11; `aml` has no Time channel to begin with.
3. **Transform.** `arcsinh(x / 150)`, the usual cofactor for fluorescence data, applied to every channel
   including scatter. Values may be **negative** — that is expected for compensated data, not an error.
4. **Subsample.** 2,000 cells per file, drawn uniformly at random **without** replacement, using
   `numpy.random.default_rng(seed)`. Files with fewer than 2,000 events are padded by resampling their own
   cells. No compensation matrix was applied (none of the three datasets ships one in its FCS files).
5. **Stack** into one array per dataset, indexed by subject and by sample slot, recording labels, channel
   names, source filenames and acquisition dates alongside.

**Why 2,000 cells is enough.** This was checked rather than assumed: quantile features computed from 500,
1,000 and 2,000 cells per file all reproduce the full-data classification result within noise (AML
AUC 0.972 vs 0.967 at 30 training patients; FlowCAP-IV 0.642 vs 0.635 at the full cohort). The 448 million
cells were doing no more work than 2,000 per file.

## File structure

```python
import numpy as np

d = np.load("aml_cells_n2000_seed0.npz", allow_pickle=True)
cells = d["cells"]      # (359, 8, 2000, 7) float32 — the measurements
y     = d["y"]          # (359,)            int64   — the label, one per subject
print(d["axes"], d["channels"])
```

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `cells` | `float32` | (subjects, samples, 2000, channels) | `arcsinh(x/150)` values. Axis 0 = subject, 1 = sample slot, 2 = cell, 3 = channel |
| `y` | `int64` | (subjects,) | binary label, 1 = positive class (AML / progressed / HEU) |
| `surv` | `float64` | (subjects,) | **`fc4` only** — see below |
| `subjects` | `<U` | (subjects,) | subject identifier |
| `channels` | `<U` | (channels,) | `$PnN` channel names, aligned with axis 3 |
| `files` | `<U64` | (subjects, samples) | source FCS filename, for traceability |
| `dates` | `<U16` | (subjects, samples) | `$DATE` of acquisition — enables train-early / test-late splits |
| `axes` | `<U` | scalar | what axis 1 means for this dataset |
| `ncell`, `seed` | `int64` | scalar | subsampling parameters used |

**What axis 1 is, per dataset:**

- `aml` — tube 1…8. The same 7 detectors (`FS Lin, SS Log, FL1–FL5 Log`) in every tube, but a *different
  antibody panel* per tube, so cells are not comparable across tubes.
- `fc4` — `[stimulated, unstimulated]`, the same panel in both.
- `heu` — stimulation condition, in order `CPG, LPS, PAM, PG, PIC, R848, unstim`, same panel throughout.

The unit of analysis is the **subject** (axis 0). Splitting on anything finer — files, tubes, cells — puts
the same patient in train and test and will produce inflated results.

## Labels

**`aml`** — 43 AML vs 316 healthy subjects (12% prevalence), from `AML.csv`. Constant within a subject
across all 8 tubes.

**`heu`** — 20 HIV-exposed-uninfected vs 24 unexposed infants, from `HEUvsUE.csv`. At n = 44 a 30%
held-out test set is only 13 patients, which cannot resolve anything; use repeated stratified k-fold
cross-validation over the whole cohort instead.

**`fc4`** — `y` is the observed clinical status: 1 = progression to AIDS or death (79 subjects), 0 = no
progression (304). `surv` is the accompanying **right-censored** time in days: the time to onset of AIDS
for the 79 who progressed, but the *time to last evaluation* for the other 304. Nine values are negative,
meaning the sample was taken shortly after AIDS diagnosis.

> **`surv` is not a regression target.** Fitting a regressor on it trains on follow-up duration as if it
> were an event time. A naive ridge regression reaches Spearman +0.37 on it, while the correlation *within
> the 79 real events* is −0.13 — it is predicting censoring, not biology. Use it as an *evaluation* metric
> instead: treat your classifier's predicted probability as a risk score and report its concordance index
> against `(surv, y)`, which is the framework the original challenge used (Cox proportional hazards and a
> log-rank test).

The original FlowCAP-IV challenge released labels for 191 subjects and withheld 192. That split is
recoverable from the dataset's `MetaDataTrain.csv` if you want results comparable to the 2014 entries.

## Channels and markers

- **`aml`** carries no antibody names at all — `$PnS` is empty, so the channels are just `FS Lin, SS Log,
  FL1 Log … FL5 Log`. Nothing here can be interpreted biologically. This costs the models nothing (none of
  them uses column names) but it does make feature importances uninterpretable.
- **`fc4`** is the opposite: its dataset page ships a `FlowCAPchannels.csv` naming 13 of the channels —
  `B515-A = IFNG, G780-A = TNFA, G710-A = CD4, G660-A = CD27, G610-A = CD107A, G560-A = CD154,
  R780-A = CD3, R710-A = CCR7, R660-A = IL2, V800-A = CD8, V585-A = CD57, V545-A = CD45RO,
  V450-A = VIVID/CD14`, plus `FSC-A, FSC-H, SSC-A`. Five channels are marked *"Not used"* in the original
  panel (`B710-A, V705-A, V655-A, V605-A, V565-A`) and are **still present in the cube** — dropping them
  is a reasonable thing to try.
- **`heu`** carries detector names only (`FSC-A, SSC-A, FITC-A, PE-A, PerCP-Cy5-5-A, PE-Cy7-A, APC-A,
  APC-Cy7-A, Pacific Blue-A, Alex 700-A`).

## Known limitations

- **Rare populations are gone.** At 2,000 cells per file, anything below roughly 0.5% prevalence is
  represented by fewer than ten cells. These cubes are unsuitable for rare-subset questions.
- **The cells-per-patient axis only runs downward** from 2,000. Going higher means regenerating the cubes
  from the source archive (about 15 s per dataset).
- **A fixed subsample hides its own variance.** With a single draw you cannot tell how much of a result
  depends on which cells happened to be picked; a second draw is needed to measure that.
- **Uniform, not density-dependent, subsampling.** Some published pipelines (e.g. SPADE's) downsample by
  density to deliberately preserve rare populations; these cubes do not.
- **No compensation and no per-dataset transform tuning.** One cofactor (150) for everything.

## One rule for using these cubes

Any representation you compute **per file** — per-channel quantiles, threshold-based phenotype
proportions — cannot leak, because each sample is summarised in isolation. Anything **fitted across
samples** — a clustering, a cell-level classifier — must be re-fitted on the training subjects *inside
every cross-validation fold*. Fitting it once on all subjects quietly shows the test patients to the
featurisation and inflates exactly the small-sample results this project is about.

## Attribution

The underlying data are public and remain the property of their depositors. Cite the source studies:

- Aghaeepour N. et al., *Critical assessment of automated flow cytometry data analysis techniques.*
  Nature Methods 10:228–238 (2013) — source of the AML and HEU datasets.
- Aghaeepour N. et al., *A benchmark for evaluation of algorithms for identification of cellular
  correlates of clinical outcomes.* Cytometry A 89:16–21 (2016) — the FlowCAP-III/IV benchmark.

Please also cite the FlowRepository accessions (`FR-FCM-ZZYA`, `FR-FCM-ZZ99`, `FR-FCM-ZZZU`) directly.
