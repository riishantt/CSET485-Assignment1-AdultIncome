"""Builds CSET485_Assignment1_AdultIncome.ipynb from the cells below.

Run:  python build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace CSET485_Assignment1_AdultIncome.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# CSET485 – AI and Society · Assignment 1

**Name:** Rishant Ranjan  **Enrollment No.:** S24BCAU0036

**Dataset:** Adult Census Income (roll number ends in 6)  **Random seed:** 36 (last three digits, 036)

This notebook produces every number, table and chart used in the PDF report. Run it top to bottom
(`Kernel → Restart & Run All`). Outputs are written to `results/` (CSV tables and `numbers.json`)
and `figures/` (PNG charts).

Target: `y = 1` if income > $50K, else `0`.
""")

code(r"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

SEED = 36
DATA = Path("data")
RES = Path("results"); RES.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
N = {}  # every number quoted in the report is stored here and saved to results/numbers.json

# chart styling (validated categorical slots 1 and 2, light mode, for print)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})
""")

md(r"""
## 0. Data loading

The UCI folder ships two files: `adult.data` (32,561 rows) and `adult.test` (16,281 rows). The test file
has a junk first line (`|1x3 Cross validator`) and labels ending in a full stop (`>50K.`). I join both files
and make my own 70/30 split with seed 36, as the brief asks. Missing values are coded as `?`.
""")

code(r"""
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"]
train_raw = pd.read_csv(DATA / "adult.data", names=COLS, skipinitialspace=True, na_values="?")
test_raw = pd.read_csv(DATA / "adult.test", names=COLS, skipinitialspace=True, na_values="?", skiprows=1)
raw = pd.concat([train_raw, test_raw], ignore_index=True)
raw["income"] = raw["income"].str.rstrip(".")

N["raw_rows"] = len(raw)
N["raw_pos_rate"] = (raw.income == ">50K").mean()
print(raw.shape)
print(raw.income.value_counts())
print("\nMissing values per column:")
print(raw.isna().sum()[lambda s: s > 0])
""")

md(r"""
## 1. Data preprocessing

**Missing values.** Only three columns have `?`: `workclass` (2,799), `occupation` (2,809) and
`native_country` (857). They overlap heavily (almost every row missing `workclass` also misses `occupation`).
I drop rows with any missing value. Imputing a job title would invent the variable the model leans on most,
and the loss is small (see the count below).

**Duplicates.** I drop exact duplicate rows, including identical `fnlwgt`, because these are almost certainly
double entries.

**Dropped columns.**
- `fnlwgt` is a survey sampling weight, not a trait of the person.
- `education` is the same information as `education_num` (one-to-one mapping).
- `relationship` mostly recodes marital status × sex (`Husband` = married man, `Wife` = married woman). Keeping it
  next to `married` and `male` would split one effect across collinear columns.

**Rare categories.** `workclass = Without-pay` has 21 rows and only 2 high earners, which causes quasi-separation
in the logit, so I remove those rows. `occupation = Armed-Forces` (14 rows) is merged into `Protective-serv`.

**Categorical encoding.**
- `sex` → `male` (1 = Male).
- `marital_status` → `married` (1 = Married-civ-spouse or Married-AF-spouse).
- `native_country` → `native_us` (1 = United-States; 91% of rows, and 40 other countries are too thin to estimate).
- `race`, `workclass`, `occupation` → one-hot dummies with a reference level dropped
  (White, Private, Adm-clerical).

**Outliers and skew.** `capital_gain` is 0 for 92% of people and top-coded at 99,999 (229 rows);
`capital_loss` is 0 for 95%. I apply `log1p` to both instead of deleting rows, because the large values are real
and carry signal. `age` (17–90, 90 is top-coded) and `hours_per_week` (1–99) are kept as is; the IQR counts below
show how many points sit outside 1.5×IQR, and none of them are data errors.

**Scaling.** The five numeric features are z-scored with the **training-set** mean and SD, then the same
transform is applied to the test rows. Logistic coefficients for these features are therefore "change in log-odds
per one SD".
""")

code(r"""
df = raw.dropna().copy()
N["rows_missing_dropped"] = len(raw) - len(df)
n_before = len(df)
df = df.drop_duplicates()
N["rows_duplicates_dropped"] = n_before - len(df)
N["rows_without_pay_dropped"] = int((df.workclass == "Without-pay").sum())
df = df[df.workclass != "Without-pay"]
df["occupation"] = df["occupation"].replace({"Armed-Forces": "Protective-serv"})

df["y"] = (df.income == ">50K").astype(int)
df["male"] = (df.sex == "Male").astype(int)
df["married"] = df.marital_status.isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
df["native_us"] = (df.native_country == "United-States").astype(int)
df["log_cap_gain"] = np.log1p(df.capital_gain)
df["log_cap_loss"] = np.log1p(df.capital_loss)
df = df.reset_index(drop=True)

N["clean_rows"] = len(df)
N["clean_pos_rate"] = df.y.mean()
N["capgain_zero_share"] = (df.capital_gain == 0).mean()
N["capgain_topcoded"] = int((df.capital_gain == 99999).sum())
N["caploss_zero_share"] = (df.capital_loss == 0).mean()

# IQR outlier counts (reported, not removed)
iqr_rows = []
for c in ["age", "hours_per_week", "capital_gain", "capital_loss", "education_num"]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    out = ((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum()
    iqr_rows.append({"feature": c, "min": df[c].min(), "q1": q1, "median": df[c].median(), "q3": q3,
                     "max": df[c].max(), "n_outside_1.5IQR": int(out), "pct": round(100 * out / len(df), 2)})
iqr = pd.DataFrame(iqr_rows)
iqr.to_csv(RES / "preprocessing_outliers.csv", index=False)
print(f"Dropped for missing values: {N['rows_missing_dropped']}, duplicates: {N['rows_duplicates_dropped']}, "
      f"Without-pay: {N['rows_without_pay_dropped']}  ->  final rows: {N['clean_rows']}")
print(f"Share earning >50K: {N['clean_pos_rate']:.4f}")
iqr
""")

code(r"""
NUM = ["age", "education_num", "hours_per_week", "log_cap_gain", "log_cap_loss"]
X = pd.concat([
    df[NUM + ["male", "married", "native_us"]],
    pd.get_dummies(df.race, prefix="race").drop(columns="race_White"),
    pd.get_dummies(df.workclass, prefix="wc").drop(columns="wc_Private"),
    pd.get_dummies(df.occupation, prefix="occ").drop(columns="occ_Adm-clerical"),
], axis=1).astype(float)
y = df.y

# 70/30 split, seed 36, stratified so both halves keep the 24.8% positive rate
idx_tr, idx_te = train_test_split(df.index, test_size=0.30, random_state=SEED, stratify=y)
mu, sd = X.loc[idx_tr, NUM].mean(), X.loc[idx_tr, NUM].std()
Xs = X.copy()
Xs[NUM] = (X[NUM] - mu) / sd
Xc = sm.add_constant(Xs)

N["n_train"], N["n_test"] = len(idx_tr), len(idx_te)
N["n_features"] = X.shape[1]
N["sd_raw"] = sd.round(4).to_dict()
print(f"Train {len(idx_tr)}, test {len(idx_te)}, features {X.shape[1]}")
print("Training-set SD used for scaling:\n", sd.round(3))
""")

md(r"""
## Part 1 — Build

- **Model A**: OLS on the 0/1 target (a linear probability model).
- **Model B**: logistic regression.

Both are fit with `statsmodels` on the training split, without regularisation, using the same 29 features.
""")

code(r"""
Xtr, Xte, ytr, yte = Xc.loc[idx_tr], Xc.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
model_a = sm.OLS(ytr, Xtr).fit()
model_b = sm.Logit(ytr, Xtr).fit(disp=0)

pred_a = model_a.predict(Xte)          # continuous score, can leave [0, 1]
prob_b = model_b.predict(Xte)          # probability in (0, 1)

median_a = float(np.median(pred_a))
flag_a = (pred_a >= median_a).astype(int)
flag_b = (prob_b >= 0.5).astype(int)

cm_ab = pd.crosstab(pd.Series(flag_a.values, name="Model A (≥ median)"),
                    pd.Series(flag_b.values, name="Model B (P ≥ 0.5)"))
cm_ab.index = ["A: low risk", "A: high risk"]; cm_ab.columns = ["B: low risk", "B: high risk"]
mismatch = int((flag_a != flag_b).sum())

N.update({
    "p1_median_a": median_a, "p1_mean_a": float(pred_a.mean()),
    "p1_a_below0": int((pred_a < 0).sum()), "p1_a_above1": int((pred_a > 1).sum()),
    "p1_flag_a": int(flag_a.sum()), "p1_flag_b": int(flag_b.sum()),
    "p1_mismatch": mismatch, "p1_mismatch_pct": mismatch / len(yte),
    "p1_a_hi_b_lo": int(((flag_a == 1) & (flag_b == 0)).sum()),
    "p1_a_lo_b_hi": int(((flag_a == 0) & (flag_b == 1)).sum()),
    "p1_both_hi": int(((flag_a == 1) & (flag_b == 1)).sum()),
    "p1_both_lo": int(((flag_a == 0) & (flag_b == 0)).sum()),
    "p1_test_pos_rate": float(yte.mean()),
    "p1_acc_a": accuracy_score(yte, flag_a), "p1_acc_b": accuracy_score(yte, flag_b),
    "p1_r2_a": float(model_a.rsquared), "p1_pseudo_r2_b": float(model_b.prsquared),
    "p1_logit_p_at_median_a": float(prob_b[(pred_a - median_a).abs().idxmin()]),
    "p1_corr_ab": float(np.corrcoef(pred_a, prob_b)[0, 1]),
})
# Where does Model A put the people Model B flags?
N["p1_b_flagged_min_a_score"] = float(pred_a[flag_b == 1].min())
# Precision of each rule against the real label
N["p1_prec_a"] = float(yte[flag_a == 1].mean()); N["p1_prec_b"] = float(yte[flag_b == 1].mean())
N["p1_recall_a"] = float(flag_a[yte == 1].mean()); N["p1_recall_b"] = float(flag_b[yte == 1].mean())

cm_ab.to_csv(RES / "part1_confusion_A_vs_B.csv")
print(f"Median Model A prediction: {median_a:.4f}")
print(f"Model A predictions < 0: {N['p1_a_below0']},  > 1: {N['p1_a_above1']}")
print(f"Flagged high risk: A = {N['p1_flag_a']}, B = {N['p1_flag_b']}")
print(f"Mismatched: {mismatch} of {len(yte)} ({100*mismatch/len(yte):.2f}%)")
print(f"Accuracy vs real label: A = {N['p1_acc_a']:.4f}, B = {N['p1_acc_b']:.4f}")
print(f"Precision: A = {N['p1_prec_a']:.4f}, B = {N['p1_prec_b']:.4f};  Recall: A = {N['p1_recall_a']:.4f}, B = {N['p1_recall_b']:.4f}")
cm_ab
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bins = np.linspace(-0.4, 1.3, 70)
ax.hist(pred_a, bins=bins, color=BLUE, alpha=0.75, label="Model A (OLS) prediction", edgecolor="white", linewidth=0.4)
ax.hist(prob_b, bins=bins, color=ORANGE, alpha=0.55, label="Model B (logit) probability", edgecolor="white", linewidth=0.4)
ax.axvline(median_a, color=BLUE, lw=1.6, ls="--")
ax.axvline(0.5, color=ORANGE, lw=1.6, ls="--")
ax.text(median_a - 0.015, ax.get_ylim()[1] * 0.92, f"A median = {median_a:.3f}", color=INK, fontsize=8, ha="right")
ax.text(0.515, ax.get_ylim()[1] * 0.60, "B cut-off = 0.5", color=INK, fontsize=8)
ax.set_xlabel("Predicted value on the test set"); ax.set_ylabel("People")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Two cut-offs on two score distributions", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIG / "part1_prediction_distributions.png"); plt.show()
""")

md(r"""
## Part 2.1 — Simpson's paradox hunt

Split variable: **marital status** (`married` = 1 vs 0). Key predictor: **`male`**.

I fit the logit on the full cleaned dataset (all 45,154 rows; `married` is a covariate in this model), then on
each subgroup (where `married` is constant, so it drops out). I also fit one pooled model with a `male × married`
interaction to test whether the two subgroup slopes really differ.
""")

code(r"""
def logit(mask, drop=()):
    return sm.Logit(y[mask], Xc.loc[mask].drop(columns=list(drop))).fit(disp=0)

all_rows = pd.Series(True, index=df.index)
m_full = logit(all_rows)
m_unmar = logit(df.married == 0, drop=["married"])
m_mar = logit(df.married == 1, drop=["married"])

def row(name, m, n):
    ci = m.conf_int().loc["male"]
    return {"model": name, "n": n, "coef_male": m.params["male"], "ci_low": ci[0], "ci_high": ci[1],
            "p_value": m.pvalues["male"], "odds_ratio": np.exp(m.params["male"])}

simpson = pd.DataFrame([
    row("Full dataset", m_full, len(df)),
    row("Subgroup A: not married", m_unmar, int((df.married == 0).sum())),
    row("Subgroup B: married", m_mar, int((df.married == 1).sum())),
])
# formal test of the difference: interaction model
Xi = Xc.copy(); Xi["male_x_married"] = Xi.male * Xi.married
m_int = sm.Logit(y, Xi).fit(disp=0)
N["s_interaction_coef"] = float(m_int.params["male_x_married"])
N["s_interaction_ci"] = m_int.conf_int().loc["male_x_married"].tolist()
N["s_interaction_p"] = float(m_int.pvalues["male_x_married"])
for _, r in simpson.iterrows():
    k = {"Full dataset": "full", "Subgroup A: not married": "unmar", "Subgroup B: married": "mar"}[r.model]
    N[f"s_{k}_coef"], N[f"s_{k}_ci"], N[f"s_{k}_p"], N[f"s_{k}_or"], N[f"s_{k}_n"] = (
        r.coef_male, [r.ci_low, r.ci_high], r.p_value, r.odds_ratio, int(r.n))
simpson.to_csv(RES / "part2_1_simpson_coefficients.csv", index=False)

# raw (unadjusted) rates: the paradox is visible even without a model
raw_rates = df.groupby(["married", "sex"]).y.agg(["mean", "size"]).rename(columns={"mean": "share_>50K", "size": "n"})
overall = df.groupby("sex").y.mean()
N["s_raw_overall"] = overall.to_dict()
N["s_raw_by_group"] = {f"{'married' if m else 'unmarried'}_{s}": float(v) for (m, s), v in raw_rates["share_>50K"].items()}
N["s_share_married_by_sex"] = df.groupby("sex").married.mean().to_dict()
N["s_unadjusted_male_coef"] = float(sm.Logit(y, sm.add_constant(df[["male"]])).fit(disp=0).params["male"])
raw_rates.to_csv(RES / "part2_1_raw_rates.csv")

print(simpson.round(4).to_string(index=False))
print(f"\nInteraction male×married: {N['s_interaction_coef']:.4f}, 95% CI {np.round(N['s_interaction_ci'],4)}, p = {N['s_interaction_p']:.2e}")
print("\nRaw share earning >50K:\n", raw_rates.round(4))
print("\nShare married, by sex:", {k: round(v, 4) for k, v in N["s_share_married_by_sex"].items()})
""")

code(r"""
# other predictors that move across the split (for context in the report)
comp = pd.DataFrame({"full": m_full.params, "not_married": m_unmar.params, "p_not_married": m_unmar.pvalues,
                     "married": m_mar.params, "p_married": m_mar.pvalues}).drop(index=["const"])
comp.round(4).to_csv(RES / "part2_1_all_coefficients_by_subgroup.csv")
comp.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.4, 2.2))
labels = ["Full dataset", "Not married", "Married"]
for i, (m, c) in enumerate(zip([m_full, m_unmar, m_mar], [MUTED, BLUE, ORANGE])):
    lo, hi = m.conf_int().loc["male"]; b = m.params["male"]
    ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(b, i, "o", color=c, ms=7, mec="white", mew=1.5)
    ax.text(hi + 0.03, i, f"{b:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8, color=INK)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(3)); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlim(-0.6, 1.25); ax.set_xlabel("Logit coefficient on male (95% CI)")
ax.set_title("The sign of the sex effect depends on marital status", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIG / "part2_1_simpson.png"); plt.show()
""")

md(r"""
## Part 2.2 — Omitted-variable bias

Dropped variable: **`married`**. Marriage is linked to sex (in this sample 87% of married people are men, see
below), to age, and to income, so it is the obvious candidate confounder. I fit the full logit and the reduced
logit on all 45,154 rows and compare coefficients.

`% change = (reduced − full) / |full| × 100`.
""")

code(r"""
m_red = sm.Logit(y, Xc.drop(columns="married")).fit(disp=0)
ovb = pd.DataFrame({"full_model": m_full.params, "without_married": m_red.params}).drop(index=["const", "married"])
ovb["abs_change"] = ovb.without_married - ovb.full_model
ovb["pct_change"] = 100 * ovb.abs_change / ovb.full_model.abs()
ovb["p_full"] = m_full.pvalues.drop(index=["const", "married"])
ovb = ovb.sort_values("pct_change", key=np.abs, ascending=False)
ovb.round(4).to_csv(RES / "part2_2_ovb_all.csv")

focus = ["male", "age", "hours_per_week", "education_num", "log_cap_gain", "race_Black", "occ_Exec-managerial"]
ovb_focus = ovb.loc[focus]
ovb_focus.round(4).to_csv(RES / "part2_2_ovb_focus.csv")
N["ovb_focus"] = ovb_focus[["full_model", "without_married", "pct_change"]].round(4).to_dict(orient="index")
N["ovb_married_coef_full"] = float(m_full.params["married"])
N["ovb_married_or_full"] = float(np.exp(m_full.params["married"]))
N["ovb_married_share_men"] = float(df.loc[df.married == 1, "male"].mean())
N["ovb_corr_married_male"] = float(df[["married", "male"]].corr().iloc[0, 1])
N["ovb_corr_married_age"] = float(df[["married", "age"]].corr().iloc[0, 1])
N["ovb_pseudo_r2_full"], N["ovb_pseudo_r2_red"] = float(m_full.prsquared), float(m_red.prsquared)
N["ovb_aic_full"], N["ovb_aic_red"] = float(m_full.aic), float(m_red.aic)
print(f"married coefficient in full model: {N['ovb_married_coef_full']:.4f} (odds ratio {N['ovb_married_or_full']:.2f})")
print(f"corr(married, male) = {N['ovb_corr_married_male']:.3f}, corr(married, age) = {N['ovb_corr_married_age']:.3f}")
print(f"share of married people who are men: {N['ovb_married_share_men']:.4f}")
print(f"pseudo-R2 full {N['ovb_pseudo_r2_full']:.4f} vs reduced {N['ovb_pseudo_r2_red']:.4f}")
ovb_focus.round(4)
""")

code(r"""
print("Largest relative shifts among all predictors:")
ovb.head(8).round(4)
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.4, 2.8))
names = {"male": "male", "age": "age (per SD)", "hours_per_week": "hours/week (per SD)",
         "education_num": "education (per SD)", "log_cap_gain": "log capital gain (per SD)",
         "race_Black": "race: Black", "occ_Exec-managerial": "occ: Exec-managerial"}
ypos = np.arange(len(focus))
ax.barh(ypos - 0.2, ovb_focus.full_model, height=0.36, color=BLUE, label="Full model")
ax.barh(ypos + 0.2, ovb_focus.without_married, height=0.36, color=ORANGE, label="married dropped")
for i, f in enumerate(focus):
    v = max(ovb_focus.full_model[f], ovb_focus.without_married[f], 0)
    ax.text(v + 0.03, i, f"{ovb_focus["pct_change"][f]:+.0f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels([names[f] for f in focus]); ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.8); ax.set_xlabel("Logit coefficient")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
ax.set_title("Dropping married inflates the male coefficient almost tenfold", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIG / "part2_2_ovb.png"); plt.show()
""")

md(r"""
## Part 2.3 — Adversarial subset

Using Model B on the test set: confident false positives (P ≥ 0.75, y = 0) and confident false negatives
(P ≤ 0.25, y = 1). I sample 50 of them with seed 36 and compare them with the full test set.
""")

code(r"""
T = df.loc[idx_te].copy()
T["p"] = prob_b
conf_fp = (T.p >= 0.75) & (T.y == 0)
conf_fn = (T.p <= 0.25) & (T.y == 1)
wrong = conf_fp | conf_fn
adv = T[wrong].sample(50, random_state=SEED)
adv.to_csv(RES / "part2_3_adversarial_50_rows.csv")

N.update({"adv_total": int(wrong.sum()), "adv_pct": float(wrong.mean()),
          "adv_fp_total": int(conf_fp.sum()), "adv_fn_total": int(conf_fn.sum()),
          "adv_n_hi_conf_pos": int((T.p >= 0.75).sum()), "adv_n_hi_conf_neg": int((T.p <= 0.25).sum())})

def profile(d):
    return pd.Series({
        "n": len(d), "mean age": d.age.mean(), "age < 30": (d.age < 30).mean(), "age ≥ 50": (d.age >= 50).mean(),
        "female": (d.male == 0).mean(), "white": (d.race == "White").mean(), "black": (d.race == "Black").mean(),
        "married": d.married.mean(), "education ≤ 10 yrs (HS/some college)": (d.education_num <= 10).mean(),
        "education ≥ 13 yrs (bachelor+)": (d.education_num >= 13).mean(), "hours > 40": (d.hours_per_week > 40).mean(),
        "capital gain = 0": (d.capital_gain == 0).mean(), "capital gain or loss > 0": ((d.capital_gain > 0) | (d.capital_loss > 0)).mean(),
        "actual >50K": d.y.mean(), "mean predicted P": d.p.mean()})

adv_fn, adv_fp = adv[adv.y == 1], adv[adv.y == 0]
prof = pd.DataFrame({"full test set": profile(T), "all confident errors": profile(T[wrong]),
                     "sample of 50": profile(adv), "  of which FN": profile(adv_fn), "  of which FP": profile(adv_fp)})
prof.round(3).to_csv(RES / "part2_3_profile.csv")
N["adv_profile"] = prof.round(4).to_dict()
print(f"Confident wrong predictions: {N['adv_total']} of {len(T)} ({100*N['adv_pct']:.2f}%)  "
      f"[FP {N['adv_fp_total']}, FN {N['adv_fn_total']}]")
print(f"FN rate inside P ≤ 0.25: {N['adv_fn_total']}/{N['adv_n_hi_conf_neg']};  FP rate inside P ≥ 0.75: {N['adv_fp_total']}/{N['adv_n_hi_conf_pos']}")
prof.round(3)
""")

code(r"""
print("Occupation mix of the 50 vs test set:")
pd.DataFrame({"sample_50": adv.occupation.value_counts(normalize=True),
              "test_set": T.occupation.value_counts(normalize=True)}).fillna(0).round(3).sort_values("sample_50", ascending=False)
""")

md(r"""
Two checks on *why* the model fails here:

1. **Calibration.** If the model is calibrated, a band with average P ≈ 0.16 should contain about 16% positives.
   Confident false negatives would then be the expected tail of an uncertain band, not a bug.
2. **Capital gain is not monotone.** The model enters `log(1 + capital_gain)` as one straight line in log-odds.
   The table below shows what the raw data says for each band of capital gain.
""")

code(r"""
bands = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0001]
cal = T.groupby(pd.cut(T.p, bands, right=False)).agg(n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean"))
cal.round(4).to_csv(RES / "part2_3_calibration.csv")
N["adv_calibration"] = {str(k): v for k, v in cal.round(4).to_dict(orient="index").items()}

cg_band = pd.cut(df.capital_gain, [-1, 0, 3000, 5100, 7000, 100000],
                 labels=["0", "1–3,000", "3,001–5,100", "5,101–7,000", "> 7,000"])
cg = df.groupby(cg_band).agg(n=("y", "size"), share_over_50K=("y", "mean"))
# what the fitted logit adds to the log-odds for an example gain inside each band, versus a gain of 0
cg["example_gain"] = [0, 2000, 4000, 6000, 15000]
cg["model_logit_shift_vs_0"] = model_b.params["log_cap_gain"] * np.log1p(cg.example_gain) / sd["log_cap_gain"]
cg.round(4).to_csv(RES / "part2_3_capital_gain_bands.csv")
N["adv_capgain_bands"] = {str(k): v for k, v in cg.round(4).to_dict(orient="index").items()}
N["adv_logit_coef_capgain"] = float(model_b.params["log_cap_gain"])
N["adv_logit_coef_married"] = float(model_b.params["married"])
N["adv_logit_coef_edu"] = float(model_b.params["education_num"])
N["adv_fn_mean_p"] = float(adv_fn.p.mean())
print(cal.round(3)); print(); print(cg.round(3))
""")

code(r"""
# Does a non-linear capital-gain encoding fix the confident false positives?
def add_cg_bins(Xframe, idx):
    out = Xframe.copy()
    g = df.loc[idx, "capital_gain"]
    out["cg_1_3000"] = ((g > 0) & (g <= 3000)).astype(float)
    out["cg_3001_7000"] = ((g > 3000) & (g <= 7000)).astype(float)
    out["cg_gt_7000"] = (g > 7000).astype(float)
    return out.drop(columns="log_cap_gain")

model_b2 = sm.Logit(ytr, add_cg_bins(Xtr, idx_tr)).fit(disp=0, method="bfgs", maxiter=2000)
prob_b2 = model_b2.predict(add_cg_bins(Xte, idx_te))
fp2 = ((prob_b2 >= 0.75) & (yte == 0)); fn2 = ((prob_b2 <= 0.25) & (yte == 1))
N["adv_bins_fp"], N["adv_bins_fn"] = int(fp2.sum()), int(fn2.sum())
N["adv_bins_acc"] = accuracy_score(yte, (prob_b2 >= 0.5).astype(int))
N["adv_fp_fixed_in_sample"] = int((prob_b2.loc[adv_fp.index] < 0.75).sum())
print(f"With capital-gain bins: confident FP {N['adv_bins_fp']} (was {N['adv_fp_total']}), "
      f"confident FN {N['adv_bins_fn']} (was {N['adv_fn_total']}), accuracy {N['adv_bins_acc']:.4f} (was {N['p1_acc_b']:.4f})")
print(f"Of the {len(adv_fp)} confident FPs in the sample of 50, {N['adv_fp_fixed_in_sample']} drop below 0.75 with the binned model")
""")

md(r"""
## Part 2.4 — Fairness trade-off

Sensitive variable: **sex**. Group A = Female, Group B = Male. Metrics are computed on the test set from Model B's
probabilities at thresholds 0.05, 0.15, …, 0.95.

- DP = P(Ŷ=1 | Female) / P(Ŷ=1 | Male)
- EO = min(TPR_F / TPR_M, FPR_F / FPR_M)
- PP = P(Y=1 | Ŷ=1, Female) / P(Y=1 | Ŷ=1, Male)

A ratio is left blank (NaN) if its denominator is zero.
""")

code(r"""
def rates(yt, yp):
    tp = ((yp == 1) & (yt == 1)).sum(); fp = ((yp == 1) & (yt == 0)).sum()
    pos, neg = (yt == 1).sum(), (yt == 0).sum()
    return {"sel": yp.mean(), "tpr": tp / pos if pos else np.nan, "fpr": fp / neg if neg else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) else np.nan}

def ratio(a, b):
    return a / b if (b and not np.isnan(b)) else np.nan

fem, mal = (T.male == 0).values, (T.male == 1).values
yv, pv = T.y.values, T.p.values
rows = []
for t in np.round(np.arange(0.05, 1.0, 0.10), 2):
    yp = (pv >= t).astype(int)
    rf, rm = rates(yv[fem], yp[fem]), rates(yv[mal], yp[mal])
    tpr_r, fpr_r = ratio(rf["tpr"], rm["tpr"]), ratio(rf["fpr"], rm["fpr"])
    rows.append({"threshold": t,
                 "DP": ratio(rf["sel"], rm["sel"]),
                 "EO": np.nanmin([tpr_r, fpr_r]) if not (np.isnan(tpr_r) and np.isnan(fpr_r)) else np.nan,
                 "PP": ratio(rf["ppv"], rm["ppv"]),
                 "TPR_ratio": tpr_r, "FPR_ratio": fpr_r,
                 "sel_F": rf["sel"], "sel_M": rm["sel"], "TPR_F": rf["tpr"], "TPR_M": rm["tpr"],
                 "FPR_F": rf["fpr"], "FPR_M": rm["fpr"], "PPV_F": rf["ppv"], "PPV_M": rm["ppv"],
                 "acc_all": (yp == yv).mean(), "acc_F": (yp[fem] == yv[fem]).mean(), "acc_M": (yp[mal] == yv[mal]).mean()})
fair = pd.DataFrame(rows)
fair.round(4).to_csv(RES / "part2_4_fairness_sweep.csv", index=False)
N["fair"] = fair.round(4).to_dict(orient="records")
N["fair_base_rate_F"], N["fair_base_rate_M"] = float(yv[fem].mean()), float(yv[mal].mean())
N["fair_n_F"], N["fair_n_M"] = int(fem.sum()), int(mal.sum())
print(f"Test set: {fem.sum()} women (base rate {yv[fem].mean():.4f}), {mal.sum()} men (base rate {yv[mal].mean():.4f})")
fair[["threshold", "DP", "EO", "PP", "TPR_ratio", "FPR_ratio", "acc_all"]].round(3)
""")

code(r"""
fair[["threshold", "sel_F", "sel_M", "TPR_F", "TPR_M", "FPR_F", "FPR_M", "PPV_F", "PPV_M", "acc_F", "acc_M"]].round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.fill_between([0.8, 1.25], 0.8, 1.25, color=GRID, alpha=0.6, lw=0)
ax.text(0.82, 1.21, "within 0.8–1.25 of parity", fontsize=7.5, color=MUTED)
ax.plot([0, 1.5], [1, 1], color=MUTED, lw=0.8, ls=":"); ax.plot([1, 1], [0.76, 1.5], color=MUTED, lw=0.8, ls=":")
ok = fair.dropna(subset=["DP", "EO"])
ax.plot(ok.DP, ok.EO, color=BLUE, lw=1.2, alpha=0.6)
ax.scatter(ok.DP, ok.EO, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
for _, r in ok[ok.threshold.isin([0.05, 0.15, 0.95])].iterrows():
    ax.annotate(f"t={r.threshold:.2f}", (r.DP, r.EO), xytext=(7, -3), textcoords="offset points", fontsize=7.5, color=INK)
# the thresholds 0.25-0.85 sit in a tight cluster, so they get a zoomed inset
x0, x1, y0, y1 = 0.10, 0.31, 0.07, 0.25
ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=0.8, ls="--"))
ins = ax.inset_axes([0.56, 0.07, 0.41, 0.36])
ins.plot(ok.DP, ok.EO, color=BLUE, lw=1.2, alpha=0.6)
ins.scatter(ok.DP, ok.EO, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
offs = {0.25: (6, -2), 0.35: (6, -3), 0.45: (6, -3), 0.55: (6, -1), 0.65: (-4, 8), 0.75: (-8, -2), 0.85: (6, -4)}
for _, r in ok.iterrows():
    t = round(r.threshold, 2)
    if t in offs:
        ins.annotate(f"t={t:.2f}", (r.DP, r.EO), xytext=offs[t], textcoords="offset points", fontsize=7, color=INK,
                     ha="right" if offs[t][0] < 0 else "left")
ins.set_xlim(x0, x1); ins.set_ylim(y0, y1); ins.tick_params(labelsize=6.5)
ins.set_title("zoom: t = 0.25 to 0.85", fontsize=7.5, color=MUTED, loc="left")
for sp in ins.spines.values(): sp.set_visible(True); sp.set_color(MUTED)
ax.scatter([1], [1], marker="*", s=120, color=ORANGE, zorder=4)
ax.annotate("ideal (1, 1)", (1, 1), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_aspect("equal")
ax.set_xlabel("Demographic parity ratio (Female / Male)"); ax.set_ylabel("Equalized odds ratio (Female / Male)")
ax.set_title("DP vs EO across thresholds (Group A = Female)", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIG / "part2_4_tradeoff.png"); plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
for ax, col, lab in zip(axes, ["DP", "EO", "PP"], ["Demographic parity", "Equalized odds", "Predictive parity"]):
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.plot(fair.threshold, fair[col], color=BLUE, lw=2, marker="o", ms=4, mec="white")
    ax.set_title(lab, fontsize=9, color=INK, loc="left"); ax.set_xlabel("threshold t")
    ax.set_ylim(0, 1.5)
axes[0].set_ylabel("Female / Male ratio")
fig.tight_layout(); fig.savefig(FIG / "part2_4_metrics_vs_threshold.png"); plt.show()
""")

md(r"""
**Quantifying the trade-off.** For every pair of thresholds (t₁ → t₂) where DP gets closer to 1 while EO moves
further from 1, I compute the gain in DP, the loss in EO, and the cost W/Z. The pair with the biggest DP gain is
reported in the write-up.
""")

code(r"""
pairs = []
ok = fair.dropna(subset=["DP", "EO"]).reset_index(drop=True)
for i, a in ok.iterrows():
    for j, b in ok.iterrows():
        if i == j: continue
        dist_dp_a, dist_dp_b = abs(1 - a.DP), abs(1 - b.DP)
        dist_eo_a, dist_eo_b = abs(1 - a.EO), abs(1 - b.EO)
        if dist_dp_b < dist_dp_a and dist_eo_b > dist_eo_a:
            z = 100 * (b.DP - a.DP) / a.DP; w = 100 * (a.EO - b.EO) / a.EO
            pairs.append({"from_t": a.threshold, "to_t": b.threshold, "DP_from": a.DP, "DP_to": b.DP,
                          "DP_gain_pct": z, "EO_from": a.EO, "EO_to": b.EO, "EO_loss_pct": w,
                          "cost_W_over_Z": w / z if z else np.nan})
trade = pd.DataFrame(pairs).sort_values("DP_gain_pct", ascending=False)
trade.round(4).to_csv(RES / "part2_4_tradeoff_pairs.csv", index=False)
N["trade_pairs"] = trade.round(4).head(10).to_dict(orient="records")
trade.round(3).head(10)
""")

md(r"""
In this data DP and EO mostly rise and fall together, so the sharper conflict is between those two and
predictive parity (and accuracy). The cell below records that DP-vs-PP trade-off and the group rates at the
default 0.5 cut-off.
""")

code(r"""
f = fair.set_index("threshold")
def tradeoff(t1, t2, gain, loss):
    z = 100 * (f.loc[t2, gain] - f.loc[t1, gain]) / f.loc[t1, gain]
    w = 100 * abs(f.loc[t2, loss] - f.loc[t1, loss]) / f.loc[t1, loss]
    return {"from_t": t1, "to_t": t2, f"{gain}_from": f.loc[t1, gain], f"{gain}_to": f.loc[t2, gain], "gain_pct": z,
            f"{loss}_from": f.loc[t1, loss], f"{loss}_to": f.loc[t2, loss], "loss_pct": w, "cost": w / z,
            "acc_from": f.loc[t1, "acc_all"], "acc_to": f.loc[t2, "acc_all"]}
N["trade_dp_pp_025_005"] = tradeoff(0.25, 0.05, "DP", "PP")
N["trade_dp_eo_075_085"] = tradeoff(0.75, 0.85, "DP", "EO")
print(pd.Series(N["trade_dp_pp_025_005"]).round(4)); print(); print(pd.Series(N["trade_dp_eo_075_085"]).round(4))

yp50 = (pv >= 0.5).astype(int)
N["rates_t050"] = {"F": rates(yv[fem], yp50[fem]), "M": rates(yv[mal], yp50[mal]),
                   "acc_F": float((yp50[fem] == yv[fem]).mean()), "acc_M": float((yp50[mal] == yv[mal]).mean())}
print("\nGroup rates at t = 0.5:", pd.DataFrame({k: v for k, v in N["rates_t050"].items() if k in "FM"}).round(4), sep="\n")
""")

md(r"""
**Two real people from the test set** (used in Part 3, Q6): the woman who truly earns > $50K and the man who
does not, each with the predicted probability closest to 0.30. Their decisions at three thresholds:
""")

code(r"""
show = ["age", "sex", "marital_status", "education", "occupation", "hours_per_week", "capital_gain", "y", "p"]
woman = T[(T.male == 0) & (T.y == 1)].assign(d=lambda d: (d.p - 0.30).abs()).sort_values("d").iloc[0]
man = T[(T.male == 1) & (T.y == 0)].assign(d=lambda d: (d.p - 0.30).abs()).sort_values("d").iloc[0]
people = pd.DataFrame([woman[show], man[show]], index=["Person F", "Person M"])
for t in [0.15, 0.25, 0.35, 0.45]:
    people[f"flag@{t}"] = (people.p >= t).astype(int)
N["example_people"] = people.astype(str).to_dict(orient="index")
people
""")

code(r"""
# headline numbers the report quotes, plus the model coefficients
N["logit_coefs_train"] = model_b.params.round(4).to_dict()
N["ols_coefs_train"] = model_a.params.round(4).to_dict()
pd.DataFrame({"OLS (Model A)": model_a.params, "Logit (Model B)": model_b.params,
              "Logit p-value": model_b.pvalues}).round(4).to_csv(RES / "part1_model_coefficients.csv")

def to_py(o):
    if isinstance(o, dict): return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o): return None
    if isinstance(o, (np.integer,)): return int(o)
    return o
(RES / "numbers.json").write_text(json.dumps(to_py(N), indent=2))
print("Saved", len(N), "entries to results/numbers.json")
pd.DataFrame({"OLS (Model A)": model_a.params, "Logit (Model B)": model_b.params, "Logit p": model_b.pvalues}).round(3)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "CSET485_Assignment1_AdultIncome.ipynb")
print("notebook written with", len(cells), "cells")
