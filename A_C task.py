# %% [markdown]
# # CA5
#
# Group 37
#
# Group members:
# * Jannicke Ådalen
# * Marcus Dalaker Figenschou
# * Rikke Sellevold Vegstein


# %%
# Standard library imports
import os

import matplotlib.pyplot as plt

# Common aliases
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Scikit-learn with specific imports
from sklearn import (
    ensemble as ens,
)
from sklearn import (
    impute as imp,
)
from sklearn import (
    metrics as met,
)
from sklearn import (
    model_selection as msel,
)
from sklearn import (
    pipeline as pipe,
)
from sklearn import (
    preprocessing as prep,
)
from sklearn import (
    svm as svm,
)

# ignoring numpu stipid fiucking as shit messages
# np.seterr(all="ignore")

# %% jupyter setup
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
# Setting the styles of plots so that they have same styling throughout
sns.set_style("white")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

# Set working directory
if "CA5" in os.getcwd():
    os.chdir("..")  # Go up one level if we're in CA3

print(f"Working directory now: {os.getcwd()}")

# Load data
train_path = os.path.join("CA5", "assets", "train.csv")
test_path = os.path.join("CA5", "assets", "test.csv")

# Load data
# 1. Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# %% [markdown]
# # Data inspection and cleaning

# %% Inspecting train data
print("---TRAIN DATA---")
train_df.info()

print("---TEST DATA---")
test_df.info()
# %% Usign smple imputer


def impute_missing(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if (missing := df[numeric_cols].isnull().any()).any():
        imputer = imp.SimpleImputer(strategy="mean")
        df.loc[:, missing[missing].index] = imputer.fit_transform(
            df[missing[missing].index]
        )

    if (missing := df[categorical_cols].isnull().any()).any():
        imputer = imp.SimpleImputer(strategy="most_frequent")
        df.loc[:, missing[missing].index] = imputer.fit_transform(
            df[missing[missing].index]
        )

    return df


# Apply to train and test
train_df = impute_missing(train_df)
test_df = impute_missing(test_df)


# %% [markdown]
# We will give the long columns 12 and 13 shorter names

# %%
train_df = train_df.rename(
    columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"}
)
test_df = test_df.rename(
    columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"}
)

# Rename the storage temperature column
train_df = train_df.rename(
    columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"}
)
test_df = test_df.rename(
    columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"}
)

print("---TRAIN DATA---")
train_df.info()

print("---TEST DATA---")
test_df.info()


# %% [markdown]
# We can see that color, harvest time and avg storage temp are all object types. From both train and test data we can see that avg storage temp is missing a lot of data compared to the others. We choose therefore to drop this column, we dont have enough data to create synthetic values for this.

# %% Removing avg storage temp col

train_df = train_df.drop("Avg Storage Temp (C)", axis=1)

test_df = test_df.drop("Avg Storage Temp (C)", axis=1)

# %% Checking data again

print(train_df.info())
# %% Removing NaN

train_df = train_df.dropna()

print(train_df.info)


# %% [markdown]
# Logarithmic scaling of SHU.

train_df["Scoville Heat Units (SHU)"] = np.log1p(train_df["Scoville Heat Units (SHU)"])

print(train_df.describe())


# %% [markdown]
# We need to labels based on SHU vales, we therefore found this image online and we follow the ranges like that.


# %% Creating bins based on shu values

bins = [
    0,  # Sweet to mild
    np.log1p(5001),  # Medium
    np.log1p(15001),  # Medium-Hot
    np.log1p(100001),  # Hot
    np.log1p(350001),  # Superhot
    float("inf"),  # For anything higher (Carolina Reaper types)
]

labels = ["Sweet-Mild", "Medium", "Medium-Hot", "Hot", "Superhot"]


train_df["spiciness_labels"] = pd.cut(
    train_df["Scoville Heat Units (SHU)"], bins=bins, labels=labels, include_lowest=True
)
print(train_df[["Scoville Heat Units (SHU)", "spiciness_labels"]].head(10))

# %%

le_color = prep.LabelEncoder()
le_harvest = prep.LabelEncoder()

# Fit and transform training data
train_df["color"] = le_color.fit_transform(train_df["color"])
train_df["Harvest Time"] = le_harvest.fit_transform(train_df["Harvest Time"])

# Now transform test data using the SAME encoders (without fitting again)
test_df["color"] = le_color.transform(test_df["color"])
test_df["Harvest Time"] = le_harvest.transform(test_df["Harvest Time"])
# printing Dataframe

# %%
print(train_df[["color"]].head(10))
print(train_df[["Harvest Time"]].head(10))


print(test_df[["color"]].head(10))
print(test_df[["Harvest Time"]].head(10))

# %% [markdown]

features = train_df.columns.drop(["spiciness_labels", "Scoville Heat Units (SHU)"])
classes = train_df["spiciness_labels"].unique()

fig, axes = plt.subplots(7, 2, figsize=(15, 30))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.violinplot(
        data=train_df,
        x="spiciness_labels",
        y=feature,
        hue="spiciness_labels",
        palette="Set1",
        ax=axes[i],
        alpha=0.4,
        orient="v",
    )
    axes[i].set_title(f"Violinplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)
# Adjust layout
fig.delaxes(axes[13])
plt.tight_layout()
fig.subplots_adjust(hspace=0.5)
plt.show()

# %% [markdown]
# %%

# %% [markdown]
# Now that we have removed the categorical features color and harvest time, we can much easier plot a correlation matrix to see how the features are related to each other.


# ## Feature Correlation Matrix
features = train_df.columns.drop(["spiciness_labels"])
classes = train_df["spiciness_labels"].unique()

# Print observation counts for each category
print("Number of observations per category:")
for cls in classes:
    count = len(train_df[train_df["spiciness_labels"] == cls])
    print(f"{cls}: {count} observations")

# %%

# %% Corr matrix for all classes
samples = [train_df[train_df["spiciness_labels"] == cls][features] for cls in classes]

# Set up the figure and axes
fig, axes = plt.subplots(5, 1, figsize=(10, 50))
axes = axes.flatten()

# Plot correlation matrix for each class
for i, (df, title) in enumerate(zip(samples, classes)):
    # Add min_periods parameter here
    correlation_matrix = df.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[i])
    axes[i].set_title(f"Correlation Matrix - {title} (n={len(df)})")

plt.tight_layout()
plt.xticks(rotation=90)
fig.subplots_adjust(hspace=1)
plt.show()


# %%
# Define your features (excluding target and derived variables)
features = train_df.columns.drop(["spiciness_labels"])

# Create correlation matrix for the entire dataset
correlation_matrix = train_df[features].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Feature Engineering


# ## Feature Correlation Matrix

train_df_features = train_df.copy()
test_df_features = test_df.copy()

# %%

train_df_features["Firm_seed"] = (
    train_df_features["Firmness"] * train_df_features["Seed Count"]
)

# higher order terms
train_df_features["Pericarp_Squared"] = (
    train_df_features["Pericarp Thickness (mm)"] ** 2
)

# Try these alternative interactions
train_df_features["Capsaicin_x_Sugar"] = (
    train_df_features["Capsaicin Content"] * (train_df_features["Sugar Content"])
)
train_df_features["VitaminCxCapsaicin"] = (
    train_df_features["Vitamin C Content (mg)"] * train_df_features["Capsaicin Content"]
)

train_df_features["Thickness_x_Capsaicin"] = (
    train_df_features["Pericarp Thickness (mm)"]
    * train_df_features["Capsaicin Content"]
)


# %%

feature_engineering = train_df_features.columns.drop(
    ["spiciness_labels", "Scoville Heat Units (SHU)"]
)
classes = train_df_features["spiciness_labels"].unique()

fig, axes = plt.subplots(10, 2, figsize=(15, 30))
axes = axes.flatten()
for i, feature in enumerate(feature_engineering):
    sns.boxplot(
        data=train_df_features,
        x="spiciness_labels",
        y=feature,
        hue="spiciness_labels",
        palette="Set1",
        ax=axes[i],
    )
    axes[i].set_title(f"Boxplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)
# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.4)
plt.show()

# %%
# Option 2: Use pandas styling with more formatting options
# Option 1: Format with pandas styling
with pd.option_context("display.precision", 2, "display.width", 120):
    print(train_df_features.describe())


# %% [markdown]
# # Regression Model Implementation
# We'll implement a regression model to predict Scoville Heat Units (SHU).
# Since we're using a logarithmic scale for SHU, we need to transform it back
# for final predictions and evaluation.


# %%


X = train_df.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels"])
y = train_df["Scoville Heat Units (SHU)"]


# %% Splitting the data into train and test data with a 80/20 split
X_train, X_test, y_train, y_test = msel.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% TASK C
# ===============================
# Først: Data split for classifier
# ===============================

train_df_features = train_df.copy()
train_df_features["Scoville Heat Units (SHU)"] = np.expm1(
    train_df_features["Scoville Heat Units (SHU)"]
)
print(train_df_features["Scoville Heat Units (SHU)"].head(10))

# %%
train_df_features["is_spicy"] = (
    train_df_features["Scoville Heat Units (SHU)"] > 0
).astype(int)

X_full = train_df_features.drop(
    columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"]
)
y_full = train_df_features["is_spicy"]

# Split data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = msel.train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

# %%
# ===============================
# Train classifier (step 1)
# ===============================

param_grid_class = {
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__n_estimators": [
        100,
        200,
        500,
    ],  # Bruk n_estimators i stedet for max_iter
    "classifier__max_depth": [3, 5, 10],  # None er ikke gyldig her, så fjern det
    "classifier__min_samples_leaf": [10, 20, 30],
}

classifier_pipe = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        ("classifier", ens.GradientBoostingClassifier(random_state=42)),
    ]
)

classifier_search = msel.RandomizedSearchCV(
    classifier_pipe,
    param_distributions=param_grid_class,
    n_iter=30,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

classifier_search.fit(X_train_bin, y_train_bin)
best_classifier_pipe = classifier_search.best_estimator_

# Evaluer
y_pred_bin = best_classifier_pipe.predict(X_test_bin)
print(met.classification_report(y_test_bin, y_pred_bin))


# %%
# ===============================
# Split spicy peppers for regression
# ===============================

spicy_train = train_df_features[train_df_features["is_spicy"] == 1]
X_spicy = spicy_train.drop(
    columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"]
)
y_spicy = spicy_train["Scoville Heat Units (SHU)"]

# Fjern outliers
z_scores_spicy = np.abs(stats.zscore(X_spicy))
filter_mask_spicy = (z_scores_spicy < 3).all(axis=1)

X_spicy_clean = X_spicy[filter_mask_spicy]
y_spicy_clean = y_spicy[filter_mask_spicy]

# Transformér SHU tilbake til log-skala
y_spicy_clean = np.log1p(y_spicy_clean)

print(f"✅ Spicy peppers før rensing: {len(X_spicy)}")
print(f"✅ Spicy peppers etter rensing: {len(X_spicy_clean)}")

# %%
# Fjern ekstremt høye SHU-verdier (outliers på y)
y_threshold = np.log1p(500_000)  # eksempelgrense
mask = y_spicy_clean < y_threshold
X_spicy_clean = X_spicy_clean[mask]
y_spicy_clean = y_spicy_clean[mask]

print(f"✅ Spicy peppers etter fjerning av høye SHU: {len(X_spicy_clean)}")
# %%
# 1. Bruk kun GradientBoostingRegressor direkte
regressor = ens.GradientBoostingRegressor(random_state=42)

# 2. Sett opp pipeline
regressor_pipe = pipe.Pipeline(
    [("scaler", prep.StandardScaler()), ("regressor", regressor)]
)

# 3. Sett opp parametere for tuning
param_grid_reg = {
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__n_estimators": [100, 200, 500],
    "regressor__max_depth": [3, 5, 10],
    "regressor__min_samples_leaf": [5, 10, 20],
    "regressor__subsample": [0.8, 1.0],
    "regressor__loss": ["squared_error", "absolute_error"],
}

# 4. RandomizedSearchCV for å finne beste modell
hgb_search = msel.RandomizedSearchCV(
    regressor_pipe,
    param_distributions=param_grid_hgb,
    n_iter=50,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

# 5. Tren søk på spicy peppers
hgb_search.fit(X_spicy_clean, y_spicy_clean)

# 6. Beste pipeline
best_spicy_regressor = hgb_search.best_estimator_
# %%
# ===============================
# Validation på spicy peppers
# ===============================

X_spicy_train, X_spicy_val, y_spicy_train, y_spicy_val = msel.train_test_split(
    X_spicy_clean, y_spicy_clean, test_size=0.2, random_state=42
)

best_spicy_regressor.fit(X_spicy_train, y_spicy_train)
y_spicy_val_pred = best_spicy_regressor.predict(X_spicy_val)

# Undo log1p (ikke noe qt mer!)
y_spicy_val_true_real = np.expm1(y_spicy_val)
y_spicy_val_pred_real = np.expm1(y_spicy_val_pred)

mae_val = met.mean_absolute_error(y_spicy_val_true_real, y_spicy_val_pred_real)
r2_val = met.r2_score(y_spicy_val_true_real, y_spicy_val_pred_real)

print("=== Validation Set Performance ===")
print(f"✅ MAE: {mae_val:.2f}")
print(f"✅ R²: {r2_val:.4f}")
print("===============================")

# %%
# ===============================
# Final Kaggle Prediction
# ===============================


X_test_final = test_df_features.copy()
is_spicy_pred = best_classifier_pipe.predict(X_test_final)

X_test_spicy = X_test_final.loc[is_spicy_pred == 1]
X_test_spicy = X_test_spicy[X_spicy_clean.columns]

shu_pred_spicy_log = best_spicy_regressor.predict(
    X_test_spicy
)  # nå predikerer vi log-verdi
shu_pred_spicy_real = np.expm1(shu_pred_spicy_log)  # undo log1p

final_predictions_real = np.zeros(len(X_test_final))
final_predictions_real[is_spicy_pred == 1] = shu_pred_spicy_real
final_predictions_real[is_spicy_pred == 0] = 0.0

submission = pd.DataFrame(
    {
        "id": np.arange(len(final_predictions_real)),
        "Scoville Heat Units (SHU)": final_predictions_real,
    }
)

submission_path = os.path.join("CA5", "results", "submission_final.csv")
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission.to_csv(submission_path, index=False)

print(f"✅ Kaggle submission lagret: {submission_path}")
print(submission.head())
# %%
