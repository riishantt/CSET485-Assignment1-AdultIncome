# CSET485 AI and Society: Assignment 1

**Rishant Ranjan, S24BCAU0036**

I trained a linear probability model and a logistic regression on the UCI Adult Census Income data to predict whether a person earns more than $50K, then stress-tested them. The dataset is assigned because my roll number ends in 6. The train/test split uses seed 36, the last three digits of my roll number.

- **Report (PDF):** [`report/Rishant_Ranjan_S24BCAU0036_CSET485_Assignment1.pdf`](report/Rishant_Ranjan_S24BCAU0036_CSET485_Assignment1.pdf)
- **Notebook:** [`CSET485_Assignment1_AdultIncome.ipynb`](CSET485_Assignment1_AdultIncome.ipynb). Every number in the report comes from here.

## Main results

| Test | Result |
|---|---|
| Part 1: median-of-OLS rule vs logit at 0.5 | 4,016 of 13,547 test cases (29.64%) are flagged differently, all of them flagged by OLS only |
| 2.1 Simpson's paradox (split by marital status) | Coefficient on `male`: +0.120 overall, +0.639 for unmarried people, −0.290 for married people |
| 2.2 Omitted-variable bias (drop `married`) | Coefficient on `male` rises from 0.120 to 1.167 (+875%) |
| 2.3 Adversarial subset | 751 confident errors (5.5%). Capital gains of $1–3,000 have a 0% >$50K rate, but the model still raises the score for them |
| 2.4 Fairness (Female vs Male) | DP and EO stay below 0.6 at every threshold. Recommended t = 0.25 (PP = 0.985) |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace CSET485_Assignment1_AdultIncome.ipynb   # results/ and figures/
python report/build_pdf.py                                                                  # PDF (needs Chrome, Brave or Chromium)
```

`build_notebook.py` generates the notebook's cells. Edit that file, then run `python build_notebook.py` to regenerate the notebook.

## Layout

```
data/       UCI Adult files (adult.data, adult.test, adult.names)
results/    every table as CSV, plus numbers.json with each quoted figure
figures/    charts used in the report
report/     report.md (source), the PDF, and build_pdf.py
```

Data: Dua, D. and Graff, C. (2019), UCI Machine Learning Repository, Adult Data Set (CC BY 4.0).
