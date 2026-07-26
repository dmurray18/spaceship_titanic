# Spaceship Titanic — Binary Classification
 
Predicts whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly, using the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset.
 
## Results (on the real, complete dataset — 8,693 passengers)
 
| Model | CV Accuracy | Validation Accuracy | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 78.9% | 78.9% | 0.792 | 0.884 |
| Random Forest | 80.2% | 80.8% | 0.801 | 0.896 |
| **XGBoost (selected)** | 79.6% | **80.9%** | **0.808** | **0.900** |
 
XGBoost was selected on validation F1 and AUC, despite a marginally lower cross-validation mean than Random Forest — within normal fold-to-fold variance, and held-out performance is the more direct estimate of generalization.
 
## The debugging story
 
An earlier draft of this pipeline ran without errors but contained 5 real defects, only found by tracing the pipeline's actual output at each step rather than trusting a clean run:
 
1. **Silent data corruption** — `OrdinalEncoder` was applied before missing values were imputed. It didn't raise an error; it silently passed 420 `NaN` values through the transform.
2. **Unconverted numeric strings** — fields derived from `.str.split()` (`Group`, `CabinNum`) stayed string type instead of numeric.
3. **Unencoded categoricals** — `Deck` and `Side` were engineered but never encoded, causing `model.fit()` to fail with `could not convert string to float: 'B'`.
4. **Encoder reuse** — one `OneHotEncoder` instance was reused across two columns, silently discarding the first column's learned categories.
5. **No train/test consistency** — the original draft never applied preprocessing to a held-out test set at all.
Full technical writeup, including the fix for each: [`writeup.pdf`](./writeup.pdf) ([LaTeX source](./writeup.tex)).
 
## Methodology
 
- **Imputation**: mode for categoricals, median for numerics — both computed from training data only, to avoid test-set leakage
- **Feature engineering**: group size from `PassengerId`, cabin deck/side from `Cabin`, total spend and a `NoSpend` flag (proxy for `CryoSleep` — see writeup for why this matters)
- **Encoding**: ordinal for binary flags, one-hot (separate encoder instances, fit on train only) for multi-category fields
- **Models**: 5-fold stratified cross-validation across Logistic Regression, Random Forest, and XGBoost
## Files
 
```
spaceship_titanic.py         # full pipeline: train + test -> submission.csv (needs Kaggle's real test.csv)
generate_report_assets.py    # regenerates the charts and metrics below from train.csv alone
writeup.tex / writeup.pdf    # full technical report
eda_overview.png             # 6-panel exploratory data analysis
best_model_diagnostics.png   # confusion matrix + feature importance for XGBoost
model_comparison_final.csv   # raw metrics table
train.csv                    # Kaggle training data (8,693 labeled passengers)
requirements.txt
```
 
## How to run
 
```bash
pip install -r requirements.txt
 
# Regenerate charts + validation metrics from train.csv alone:
python generate_report_assets.py
 
# Generate an actual Kaggle submission (requires downloading the real,
# unlabeled test.csv from the competition's Data tab first):
python spaceship_titanic.py
```
 
## Note on Kaggle submission
 
`spaceship_titanic.py` is written to consume both `train.csv` and Kaggle's official `test.csv`, producing a submission-ready `submission.csv`. The metrics reported above come from an internal held-out validation split on the labeled training data — the true test of generalization is submitting to Kaggle's leaderboard, which requires downloading the real `test.csv` from the competition page (it is not included in this repo since it isn't needed to reproduce the reported metrics).
 
