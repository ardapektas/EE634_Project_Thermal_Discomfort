
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Initial settings
INPUT        = "all_subjects_roi_temperatures.csv"
ALL_SIX      = ["forehead", "cheek", "eye", "nose", "mouth", "chin"]
FEATURES     = ["cheek", "eye", "nose", "mouth"]  
LABEL        = "Sensation"
RANDOM_STATE = 42


# Load the dataset and clean the sensation labels
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

# Label correction
df[LABEL] = df[LABEL].replace({2: 1})
df = df.dropna(subset=[LABEL]).copy()          
df[LABEL] = df[LABEL].astype(int)
print(f"After label cleaning: {len(df)} rows")
print("Class distribution (-1 cooler, 0 neutral, 1 warmer):")
print(df[LABEL].value_counts().sort_index().to_string())


# Check missing temperature values
print("\nMissing region temperatures (NaN count per region):")
print(df[ALL_SIX].isna().sum().to_string())

# Estimate the importance of all six ROIs
imp_rows = df.dropna(subset=ALL_SIX)
print(f"\nFeature importance computed on {len(imp_rows)} fully-complete rows:")
Xi, yi = imp_rows[ALL_SIX].values, imp_rows[LABEL].values

rf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=RANDOM_STATE).fit(Xi, yi)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,random_state=RANDOM_STATE).fit(Xi, yi)
imp = pd.DataFrame({"region": ALL_SIX, "RF":   rf.feature_importances_, "GBDT": gb.feature_importances_}).set_index("region")
print(imp.sort_values("RF").round(4).to_string())



# Prepare the dataset 
model_df = df.dropna(subset=FEATURES).copy()
print(f"\nModeling rows (complete in {FEATURES}): {len(model_df)} " f"({len(df) - len(model_df)} dropped for missing features)")

X = model_df[FEATURES].values
y = model_df[LABEL].values

scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xz, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

model_df.to_csv("dataset_clean.csv", index=False)
np.savez("module2_data.npz", X_train=X_train, X_test=X_test,y_train=y_train, y_test=y_test,features=np.array(FEATURES))
print("\nSaved -> dataset_clean.csv  and  module2_data.npz")
