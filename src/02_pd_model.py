"""12-month probability-of-default (charge-off) logistic regression, PCA comparison, out-of-time backtest, PSI monitoring."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import CUT1, CUT2, END2, FEATURES, NUM, ROOT, load, snapshot

RNG = 42
d = load()
w1 = snapshot(d, CUT1, CUT2)          # development window: Jun 2024 - May 2025
w2 = snapshot(d, CUT2, END2)          # out-of-time window:  Jun 2025 - May 2026
train, test = train_test_split(w1, test_size=0.3, random_state=RNG, stratify=w1.y)


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "bad_rate": round(float(np.mean(y)), 4), "auc": round(auc, 4),
            "gini": round(2 * auc - 1, 4), "ks": round(ks(y, p), 4)}


def deciles(y, p):
    t = pd.DataFrame({"y": y, "p": p})
    t["decile"] = pd.qcut(t.p.rank(method="first"), 10, labels=range(1, 11))
    g = t.groupby("decile", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return g.round(4)


def hosmer_lemeshow(y, p, groups=10):
    from scipy.stats import chi2
    t = pd.DataFrame({"y": y, "p": p})
    t["g"] = pd.qcut(t.p.rank(method="first"), groups, labels=False)
    a = t.groupby("g").agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
    stat = (((a.o - a.e) ** 2) / (a.e * (1 - a.e / a.n))).sum()
    return round(float(stat), 2), float(chi2.sf(stat, groups - 2))


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


# ---- 1. Logistic regression on standardized origination features ----
scaler = StandardScaler().fit(train[NUM])
def X(df):
    x = df[FEATURES].copy().astype(float)
    x[NUM] = scaler.transform(df[NUM])
    return sm.add_constant(x)

logit = sm.Logit(train.y, X(train)).fit(disp=0)
coef = pd.DataFrame({"coef_per_sd": logit.params, "odds_ratio": np.exp(logit.params),
                     "p_value": logit.pvalues, "ci_low": np.exp(logit.conf_int()[0]),
                     "ci_high": np.exp(logit.conf_int()[1])}).round(4)
vif = {c: round(float(1 / (1 - sm.OLS(X(train)[c], X(train).drop(columns=[c])).fit().rsquared)), 2)
       for c in FEATURES}

p_train, p_test, p_oot = logit.predict(X(train)), logit.predict(X(test)), logit.predict(X(w2))

# ---- 2. Benchmarks: credit score alone, and logistic on principal components ----
score_only = sm.Logit(train.y, sm.add_constant(train[["score"]])).fit(disp=0)
pca = PCA().fit(scaler.transform(train[NUM]))
cum = np.cumsum(pca.explained_variance_ratio_)
k = int(np.searchsorted(cum, 0.80) + 1)
pca_k = PCA(n_components=k).fit(scaler.transform(train[NUM]))
pcX = lambda df: sm.add_constant(pd.concat([pd.DataFrame(pca_k.transform(scaler.transform(df[NUM])),
                                 index=df.index, columns=[f"PC{i+1}" for i in range(k)]),
                                 df[[c for c in FEATURES if c not in NUM]].astype(float)], axis=1))
pca_logit = sm.Logit(train.y, pcX(train)).fit(disp=0)
loadings = pd.DataFrame(pca_k.components_.T, index=NUM, columns=[f"PC{i+1}" for i in range(k)]).round(3)

# ---- 3. Monitoring: population stability between development and out-of-time windows ----
psi_table = {c: psi(train[c].values, w2[c].values) for c in NUM}
psi_table["model_pd"] = psi(p_train.values, p_oot.values)

hl_test = hosmer_lemeshow(test.y.values, p_test.values)
hl_oot = hosmer_lemeshow(w2.y.values, p_oot.values)

out = {
    "windows": {"development": "loans active 2024-05-31, charge-off by 2025-05-31",
                "out_of_time": "loans active 2025-05-31, charge-off by 2026-05-31"},
    "full_model": {"train": metrics(train.y, p_train), "test_in_time": metrics(test.y, p_test),
                   "out_of_time": metrics(w2.y, p_oot),
                   "mean_pd_vs_actual_oot": [round(float(p_oot.mean()), 4), round(float(w2.y.mean()), 4)],
                   "hosmer_lemeshow_test": hl_test, "hosmer_lemeshow_oot": hl_oot},
    "score_only": {"test_in_time": metrics(test.y, score_only.predict(sm.add_constant(test[["score"]]))),
                   "out_of_time": metrics(w2.y, score_only.predict(sm.add_constant(w2[["score"]])))},
    "pca_model": {"components_for_80pct_variance": k, "cumulative_variance": [round(v, 3) for v in cum],
                  "test_in_time": metrics(test.y, pca_logit.predict(pcX(test))),
                  "out_of_time": metrics(w2.y, pca_logit.predict(pcX(w2)))},
    "vif": vif,
    "psi": psi_table,
}
print(json.dumps(out, indent=1))
print(coef.to_string())
print(loadings.to_string())
print("OOT deciles\n", deciles(w2.y.values, p_oot.values).to_string())
(ROOT / "results/02_pd_model.json").write_text(json.dumps(out, indent=1))
coef.to_csv(ROOT / "results/02_logit_coefficients.csv")
loadings.to_csv(ROOT / "results/02_pca_loadings.csv")
deciles(test.y.values, p_test.values).to_csv(ROOT / "results/02_deciles_test.csv")
deciles(w2.y.values, p_oot.values).to_csv(ROOT / "results/02_deciles_oot.csv")
