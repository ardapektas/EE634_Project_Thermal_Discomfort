import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
INPUT    = "all_subjects_roi_temperatures.csv"
FEATURES = ["cheek", "eye", "nose", "mouth"]
LABEL    = "Sensation"


# Broad learning
class BroadLearning(BaseEstimator, ClassifierMixin):
    def __init__(self, n_feature_nodes=10, n_windows=6, n_enhance=41, reg=1e-3, random_state=0):
        self.n_feature_nodes = n_feature_nodes
        self.n_windows = n_windows
        self.n_enhance = n_enhance
        self.reg = reg
        self.random_state = random_state

    def _expand(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        Z = np.tanh(Xb @ self.Wf_)
        Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])
        H = np.tanh(Zb @ self.Wh_)
        return np.hstack([Z, H])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        n_feat = self.n_feature_nodes * self.n_windows
        self.Wf_ = rng.normal(0, 1, (X.shape[1] + 1, n_feat))
        Z = np.tanh(np.hstack([X, np.ones((len(X), 1))]) @ self.Wf_)
        self.Wh_ = rng.normal(0, 1, (n_feat + 1, self.n_enhance))
        A = self._expand(X)
        Y = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y)]
        self.W_ = np.linalg.solve(A.T @ A + self.reg * np.eye(A.shape[1]), A.T @ Y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[np.argmax(self._expand(X) @ self.W_, axis=1)]

# Sort by distances
def dist_group(d):
    if d <= 2.0:
        return "short"
    if d <= 4.0:
        return "medium"
    return "long"

# Learning models
def build_models():
    m = {
        "RF":   RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "GBDT": GradientBoostingClassifier(n_estimators=100, max_depth=10,random_state=RANDOM_STATE),
        "GBM":  GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE),
        "BL":   BroadLearning(random_state=RANDOM_STATE),
        "XGB":  XGBClassifier(n_estimators=180, max_depth=10, learning_rate=0.1, verbosity=0, random_state=RANDOM_STATE)
    }


    return m

# Analysis
def main():
    df = pd.read_csv(INPUT)
    df[LABEL] = df[LABEL].replace({2: 1})
    df = df.dropna(subset=[LABEL] + FEATURES).copy()
    df[LABEL] = df[LABEL].astype(int)
    df["grp"] = df["Distance"].apply(dist_group)

    print("Distance values present:", sorted(df["Distance"].unique()))
    print("\nRows per distance group:")
    print(df.groupby("grp").size().reindex(["short", "medium", "long"]).to_string())
    print("\nClass split per group (cooler/-1, neutral/0, warmer/1):")
    print(pd.crosstab(df["grp"], df[LABEL]).reindex(["short", "medium", "long"]).to_string())

    groups = ["short", "medium", "long"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {g: {} for g in groups}

    for g in groups:
        sub = df[df["grp"] == g]
        X = StandardScaler().fit_transform(sub[FEATURES].values)
        y = LabelEncoder().fit_transform(sub[LABEL].values)
        for name, model in build_models().items():
            scores = cross_val_score(clone(model), X, y, cv=cv, scoring="f1_macro")
            results[g][name] = scores.mean()

    names = list(build_models().keys())
    print("\nMacro-F1 by distance (5-fold CV):")
    print(f"{'model':>5} | " + " | ".join(f"{g:>7}" for g in groups))
    print("-" * 40)
    for nm in names:
        print(f"{nm:>5} | " + " | ".join(f"{results[g][nm]:>7.3f}" for g in groups))
    print("-" * 40)
    print(f"{'best':>5} | " + " | ".join(f"{max(results[g], key=results[g].get):>7}" for g in groups))

    pd.DataFrame(results).round(4).to_csv("distance_results.csv")
    print("\nSaved -> distance_results.csv")


if __name__ == "__main__":
    main()
