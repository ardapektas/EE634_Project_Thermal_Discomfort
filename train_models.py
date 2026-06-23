import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")        
RANDOM_STATE = 42


# Broad Learning
class BroadLearning(BaseEstimator, ClassifierMixin):
    """Feature nodes + enhancement nodes, solved by ridge pseudoinverse."""
    def __init__(self, n_feature_nodes=10, n_windows=6, n_enhance=41,
                 reg=1e-3, random_state=0):
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

# Learning models
def build_models():
    models = {
        "DT":  DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion="gini", random_state=RANDOM_STATE),
        "LR":  LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000),
        "NB":  GaussianNB(),
        "SVM": SVC(C=1.0, kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "DNN": MLPClassifier(hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion="gini", random_state=RANDOM_STATE),
        "GBDT": GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "GBM":  GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.1,random_state=RANDOM_STATE),
        "BL":  BroadLearning(random_state=RANDOM_STATE),
        "XGB": XGBClassifier(n_estimators=180, max_depth=10, learning_rate=0.1, verbosity=0, random_state=RANDOM_STATE),
    }

    return models

# Train and test
def main():
    data = np.load("module2_data.npz", allow_pickle=True)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # encode labels to 0..K-1 (needed by XGB; harmless for the rest)
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    y_train_e, y_test_e = le.transform(y_train), le.transform(y_test)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train_e, y_test_e])
    names = {0: "cooler", 1: "neutral", 2: "warmer"}   

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows\n")
    print(f"{'model':>5} | {'prec':>6} | {'recall':>6} | {'F1':>6} | {'macroF1 (5-fold CV)':>20}")
    print("-" * 60)

    for name, model in build_models().items():
        model.fit(X_train, y_train_e)
        pred = model.predict(X_test)
        p = precision_score(y_test_e, pred, average="macro", zero_division=0)
        r = recall_score(y_test_e, pred, average="macro", zero_division=0)
        f = f1_score(y_test_e, pred, average="macro", zero_division=0)
        per_class = f1_score(y_test_e, pred, average=None, zero_division=0)
        cvs = cross_val_score(clone(model), X_all, y_all, cv=cv, scoring="f1_macro")

        rows.append({"model": name, "precision": p, "recall": r, "f1_macro": f,"cv_macroF1_mean": cvs.mean(), "cv_macroF1_std": cvs.std(), **{f"f1_{names[i]}": per_class[i] for i in range(len(per_class))}})
        print(f"{name:>5} | {p:>6.3f} | {r:>6.3f} | {f:>6.3f} | "
              f"{cvs.mean():>10.3f} +/- {cvs.std():.3f}")

    res = pd.DataFrame(rows).sort_values("cv_macroF1_mean", ascending=False)
    res.round(4).to_csv("model_results.csv", index=False)
    print("\nRanked by 5-fold CV macro-F1:")
    print(res[["model", "precision", "recall", "f1_macro", "cv_macroF1_mean"]].round(3).to_string(index=False))
    print("\nSaved -> model_results.csv")


if __name__ == "__main__":
    main()
