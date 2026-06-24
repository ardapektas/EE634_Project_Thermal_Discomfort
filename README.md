# Non-Intrusive Thermal Comfort from Infrared Faces

A reproduction of **Bai et al. (2024), _"Non-intrusive personal thermal comfort modeling: A machine learning approach using infrared face recognition"_** (*Building and Environment* 247, 111033, [doi:10.1016/j.buildenv.2023.111033](https://doi.org/10.1016/j.buildenv.2023.111033)).

The goal is contact-free: instead of wearables or skin-attached sensors, an infrared camera reads facial skin temperature, and a classifier predicts whether a person would prefer the room **cooler**, **neutral** or **warmer**. Everything runs on the public [Charlotte-ThermalFace dataset](https://github.com/TeCSAR-UNCC/UNCC-ThermalFace) so the full pipeline from raw radiometric images to the final model comparison is reproducible.

A live summary page lives in [`index.html`](index.html) (see *GitHub Pages* below).

---

## What it does

The work is split into two modules, mirroring the paper.

**Module 1 — skin temperature extraction.** For each thermal image, facial landmarks define polygons over six regions of interest (forehead, cheeks, eyes, nose, mouth, chin). Raw 16-bit radiometric pixels are converted to Celsius and averaged inside each polygon, producing one temperature row per image.

**Module 2 — thermal comfort modeling.** Following the paper, forehead and chin are dropped a priori, leaving **cheeks, eyes, nose, mouth**; Random Forest and GBDT importances are also computed for comparison (see [Results](#results)). The four features are z-scored and split 80/20, then eleven classifiers are trained and compared by precision, recall, F1 and macro-F1 with 5-fold cross-validation both overall and separately for short, medium and long camera distances.

---

## Repository layout

```
.
├── index.html                        
├── README.md
├── requirements.txt
│
├── extract_roi_temperatures.py        
├── batch_extract_all.py               
├── prepare_dataset.py                  
├── train_models.py                     
├── distance_analysis.py                
│
├── all_subjects_roi_temperatures.csv   
├── dataset_clean.csv                   
├── module2_data.npz                    
├── model_results.csv                   
└── distance_results.csv                
```

---

## Installation

Python 3.9+ is recommended.

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the results

The dataset images are large and not included — download Charlotte-ThermalFace from the [official repo](https://github.com/TeCSAR-UNCC/UNCC-ThermalFace) and place the per-subject landmark/metadata CSVs (`S1.csv` … `S10.csv`) and TIFFs where the scripts expect them (paths are configured at the top of each Module-1 script).

```bash
# Module 1 — extract six ROI temperatures for every image of every subject
python batch_extract_all.py            # → all_subjects_roi_temperatures.csv

# Module 2
python prepare_dataset.py              # → dataset_clean.csv, module2_data.npz
python train_models.py                 # → model_results.csv
python distance_analysis.py            # → distance_results.csv
```

If you only want to re-run modeling, the provided `all_subjects_roi_temperatures.csv` lets you skip Module 1 entirely and start at `prepare_dataset.py`.

A note carried over from the extraction code: temperatures use the Flir TLinear scale, `°C = raw / 100 − 273.15`. The paper's printed `raw/10` overshoots; `raw/100` is what lands face pixels in the expected ~23–37 °C band, so that is the default here.

---

## Results

Eleven models, ranked by 5-fold CV macro-F1 (held-out folds):

| Rank | Model | Group       | Precision | Recall | Macro-F1 | CV macro-F1 |
|-----:|-------|-------------|----------:|-------:|---------:|------------:|
| 1 | XGB  | ensemble    | 0.739 | 0.741 | 0.740 | **0.736** |
| 2 | KNN  | traditional | 0.736 | 0.744 | 0.738 | 0.735 |
| 3 | RF   | ensemble    | 0.731 | 0.738 | 0.733 | 0.733 |
| 4 | GBDT | ensemble    | 0.733 | 0.735 | 0.734 | 0.730 |
| 5 | DNN  | traditional | 0.730 | 0.735 | 0.732 | 0.730 |
| 6 | GBM  | ensemble    | 0.729 | 0.735 | 0.731 | 0.715 |
| 7 | BL   | broad       | 0.703 | 0.712 | 0.704 | 0.704 |
| 8 | DT   | traditional | 0.705 | 0.706 | 0.705 | 0.695 |
| 9 | SVM  | traditional | 0.687 | 0.705 | 0.690 | 0.690 |
| 10 | LR  | traditional | 0.622 | 0.628 | 0.624 | 0.615 |
| 11 | NB  | traditional | 0.597 | 0.638 | 0.596 | 0.588 |

Best model by camera distance (CV macro-F1):

| Band | Range | Best | Score |
|------|-------|------|------:|
| Short  | 1.0–1.8 m | XGB | 0.789 |
| Medium | 2.2–4.0 m | RF  | 0.757 |
| Long   | 4.6–6.6 m | RF  | 0.739 |

Across every model, accuracy falls monotonically as the camera moves back — the headline distance effect from the paper holds.

### Feature importance 

The pipeline keeps cheeks, eyes, nose and mouth to match the paper, but the importances computed on this data tell a different story:

| Region | RF | GBDT |
|--------|----:|----:|
| cheek | **0.310** | **0.621** |
| eye | 0.162 | 0.070 |
| chin | 0.148 | 0.057 |
| nose | 0.147 | 0.085 |
| forehead | 0.143 | 0.113 |
| mouth | 0.091 | 0.054 |

Cheek dominates both rankings, but mouth — a *retained* feature — is the weakest in both, and forehead/chin are not clearly the bottom two. So the four-region choice here rests on the paper's precedent rather than on this run's importances.

### How this compares to the paper

**Confirmed.** The nose is the coldest region (32.87 °C here), ensemble models beat the linear baselines, the neutral class is the hardest to predict, and performance degrades with distance.

**Diverged — feature importance.** The model uses the paper's four regions, but this run's RF/GBDT importances don't reproduce the paper's ranking (cheek dominates; mouth, a kept feature, ranks lowest; forehead and chin aren't unambiguously weakest). See the table above.

**Diverged — models.** The paper reports Broad Learning and DCF as joint-best, with BL precision near 90%. Here the models sit in a tighter, lower band (macro-F1 ≈ 0.59–0.74), **XGBoost leads**, and BL is mid-pack. The most likely reasons are a stricter held-out CV protocol (vs. training-set scores), a corrected TLinear scale with reconstructed ROI polygons, and the absence of DCF in this environment. BL still trains fast and stays competitive, so its core argument, a quick alternative to deep networks, survives even though its top ranking does not.

---

## GitHub Pages

`index.html` is a self-contained page (no build step). To publish it:

1. Push this repo to GitHub.
2. Open **Settings → Pages**.
3. Under *Build and deployment*, set **Source: Deploy from a branch**, pick your branch and the `/ (root)` folder, and save.
4. The site goes live at `https://<username>.github.io/<repo>/` within a minute or two.

To keep the page at a custom path instead, move `index.html` into a `/docs` folder and select that folder as the Pages source.

---

## Citation

```bibtex
@article{bai2024nonintrusive,
  title   = {Non-intrusive personal thermal comfort modeling:
             A machine learning approach using infrared face recognition},
  author  = {Bai, Yan and Liu, Liang and Liu, Kai and Yu, Shuai and Shen, Yifan and Sun, Di},
  journal = {Building and Environment},
  volume  = {247},
  pages   = {111033},
  year    = {2024},
  doi     = {10.1016/j.buildenv.2023.111033}
}
```

Dataset: Ashrafi, R., Azarbayjani, M., & Tabkhi, H. (2022). Charlotte-ThermalFace: A fully annotated thermal infrared face dataset. *Infrared Physics & Technology*, 124, 104209.

## License

Released under the MIT License for the reproduction code. The Charlotte-ThermalFace dataset and the original paper remain under their respective licenses and terms.

---

> This is an independent reproduction for research and educational use. The numbers here reflect this implementation's choices and are not the authors' official figures.
