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

# %% [markdown]
# # Heisenbug
# There is right now apparently a "Heisenbug" which only affects macs with the M4 chips that one in the group members uses. We therefore have to set numpy to ignore all warnings to prevent RuntimeWarning messages from being displayed.
# https://github.com/numpy/numpy/issues/28687, due to this error we are quite limited to selecting our models.

# %%
np.seterr(all="ignore")

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
# %% [markdown]
# Both the test and training data er missing values, we will therefore do imputation and add values to the missing columns based on their mean values.

# %% Usign smple imputer


def impute_missing(df):
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Handle missing values in numeric columns using mean imputation
    if (missing := df[numeric_cols].isnull().any()).any():
        # Create a mean imputer for numeric columns
        imputer = imp.SimpleImputer(strategy="mean")
        # Then only impute columns that actually have missing values
        df.loc[:, missing[missing].index] = imputer.fit_transform(
            df[missing[missing].index]
        )

    # Handle missing values in categorical columns using mode imputation
    if (missing := df[categorical_cols].isnull().any()).any():
        # Create a most frequent (mode) imputer for categorical columns
        imputer = imp.SimpleImputer(strategy="most_frequent")
        # Then only impute columns that actually have missing values
        df.loc[:, missing[missing].index] = imputer.fit_transform(
            df[missing[missing].index]
        )

    return df


# Apply to train and test
train_df = impute_missing(train_df)
test_df = impute_missing(test_df)


# %% [markdown]
# Checking

# %%
print("---TRAIN DATA---")
train_df.info()

print("---TEST DATA---")
test_df.info()

# %% [markdown]
# The imputation has worked.

# %% [markdown]
# We will shorten the names of column 12 and 13 for better handling

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
# ## Statistical inpsection

# %%
print(train_df.describe())

# %% [markdown]
# We will drop storage temp, color and harvest time since these are object types and it will be complicated to interpret these

# %% Removing avg storage temp col
train_df = train_df.drop(
    columns=["Avg Storage Temp (C)", "color", "Harvest Time"], axis=1
)
test_df = test_df.drop(
    columns=["Avg Storage Temp (C)", "color", "Harvest Time"], axis=1
)

# %% [markdown]
# ## Log transformation

# %% [markdown]
# We are dealing with regression now which is sensitive to large value ranges, we can see that we have multiple features that have large value ranges such as: seed count, vitamin c, weight and moisture. Also SHU has a large value range. We will thereofre use np.log1p which applies the log(1+x). This is to prevent it from taking log of 0 which is not possible. We must also after we have made our predictions revert back to the normal SHU scale.

# %%
# Transformation of features
features_to_log = [
    "Weight (g)",
    "Seed Count",
    "Moisture Content",
    "Vitamin C Content (mg)",
]

# Apply log transformation to selected features
for feature in features_to_log:
    train_df[feature] = np.log1p(train_df[feature])
    test_df[feature] = np.log1p(test_df[feature])

# Apply log transformation to SHU in training data only
train_df["Scoville Heat Units (SHU)"] = np.log1p(train_df["Scoville Heat Units (SHU)"])


print(train_df.describe())


# %% [markdown]
# ## Labelling

# %% [markdown]
# We will add labels to SHU based on spicyness here for easier plotting and interpretation


# %% Creating bins based on shu values
bins = [
    0,  # Sweet
    np.log1p(1),  # Mild
    np.log1p(5001),  # Medium
    np.log1p(15001),  # Medium-Hot
    np.log1p(100001),  # Hot
    np.log1p(350001),  # Superhot
    float("inf"),  # For anything higher (Carolina Reaper types)
]

labels = ["Sweet", "Mild", "Medium", "Medium-Hot", "Hot", "Superhot"]


train_df["spiciness_labels"] = pd.cut(
    train_df["Scoville Heat Units (SHU)"], bins=bins, labels=labels, include_lowest=True
)
print(train_df[["Scoville Heat Units (SHU)", "spiciness_labels"]].head(10))
# %% [markdown]
# # Violinplot

# %%
features = train_df.columns.drop(["spiciness_labels", "Scoville Heat Units (SHU)"])
classes = train_df["spiciness_labels"].unique()

fig, axes = plt.subplots(6, 2, figsize=(15, 20))
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
fig.delaxes(axes[11])
plt.tight_layout()
fig.subplots_adjust(hspace=0.5)
plt.show()

# %% [markdown]
# Most of the features have a unimodal distribtuion, so we can safely use standardscaler here. Looking at mild pepeprs, this is the feature that is themost skewed , this is most likely just due to the range we have chosen. We can see here that there are very few features that shows clear separation between the classes, such as pericarp thickness, capsaicin content, sugar content and seed count. We will therefore expect that these are our most important features.


# %% [markdown]
# ## Correlation Matrix

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
# We can see here that most of the features exccept of width and avg growth temp have importance towards describing Scoville heat units. The most important features are pericarp thickness, seed count, capsaicin content and sugar content. This makes sense, the less seeds the more spicy the pepper is, more sugar equals spicier pepper and more capsaicin the spicier the pepper. Our corr matrix confirms the most important features, that we guessed from the violinplot.


# %% [markdown]
# # Task C: Two-step Analysis with GradientBoosting and RandomForest
#
# First step: Create a binary classifier to separate bell peppers (SHU=0) from spicy peppers (SHU>0)
# We need to create a binary target variable first. Before that we must convert the log-transformed SHU back to original scale for creating the binary labels. We will make another column for this.
#


# %%
original_shu = np.expm1(train_df["Scoville Heat Units (SHU)"])

# %% Create binary labels (0 for bell peppers, 1 for spicy peppers)
# Create binary feature 'is_spicy' where 1 indicates the pepper has SHU > 0 (spicy),
# and 0 indicates the pepper has no capsaicin (sweet/bell peppers)
train_df["is_spicy"] = (original_shu > 0).astype(int)

# Check distribution of bell vs spicy peppers
print("Distribution of pepper types:")
print(train_df["is_spicy"])

# %% Prepare data for binary classification
X_binary = train_df.drop(
    columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"]
)
y_binary = train_df["is_spicy"]

# Split data for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = msel.train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Create binary classifier pipeline with GradientBoostingClassifier
binary_pipe = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        ("classifier", ens.GradientBoostingClassifier(random_state=42)),
    ]
)

# Define hyperparameter grid for tuning GradientBoostingClassifier
binary_grid = {
    # Testing different numbers of trees (50, 100, 200) to find optimal ensemble size
    "classifier__n_estimators": [50, 100, 200],
    # Testing various learning rates (0.01, 0.1, 0.5) to optimize the gradient descent step size
    "classifier__learning_rate": [0.01, 0.1, 0.5],
    # Testing different tree depths (3, 5, 7) to balance complexity and generalization
    "classifier__max_depth": [3, 5, 7],
    # Testing various split thresholds (2, 5, 10) to control node creation criteria
    "classifier__min_samples_split": [2, 5, 10],
    # Testing partial (0.8) vs full (1.0) sample usage for building each tree
    "classifier__subsample": [0.8, 1.0],
}

# Optimize binary classifier
binary_cv = msel.GridSearchCV(
    binary_pipe,
    param_grid=binary_grid,
    cv=3,
    n_jobs=-1,
    scoring="f1",  # F1 score is good for potentially imbalanced classes
    return_train_score=True,
    verbose=1,
)

# Fit binary classifier
binary_cv.fit(X_train_binary, y_train_binary)

# Print best parameters
print("Best parameters for binary classifier:", binary_cv.best_params_)
print("Best binary classification score:", binary_cv.best_score_)

# Evaluate binary classifier on test set
y_pred_binary = binary_cv.predict(X_test_binary)
print("\nBinary Classification Report:")
print(met.classification_report(y_test_binary, y_pred_binary))

# %% Second step: Regression model for spicy peppers

# Filter training data to only include spicy peppers for regression
spicy_indices_train = train_df["is_spicy"] == 1
X_regression = train_df[spicy_indices_train].drop(
    columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"]
)
y_regression = train_df[spicy_indices_train]["Scoville Heat Units (SHU)"]

# %%
print(y_regression.describe())

# %% Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = msel.train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Create regression pipeline with RandomForestRegressor
regression_pipe = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        ("regressor", ens.RandomForestRegressor(random_state=42)),
    ]
)

# Define hyperparameter grid for RandomForestRegressor
regression_grid = {
    "regressor__n_estimators": [50, 100, 200, 300],
    "regressor__max_depth": [None, 10, 20, 30],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
}

# Optimize regression model
regression_cv = msel.GridSearchCV(
    regression_pipe,
    param_grid=regression_grid,
    cv=3,
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
    verbose=1,
)

# Fit regression model
regression_cv.fit(X_train_reg, y_train_reg)

# Print best parameters
print("Best parameters for regression:", regression_cv.best_params_)
print("Best regression score:", regression_cv.best_score_)

# Evaluate regression on test set
y_pred_reg = regression_cv.predict(X_test_reg)
print(f"Regression MAE: {met.mean_absolute_error(y_test_reg, y_pred_reg):.3f}")
print(f"Regression R²: {met.r2_score(y_test_reg, y_pred_reg):.3f}")

# %% Final two-step prediction pipeline
# Train final models on the full training set
# 1. Final binary classifier with GradientBoostingClassifier
final_binary_classifier = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        (
            "classifier",
            ens.GradientBoostingClassifier(
                n_estimators=binary_cv.best_params_["classifier__n_estimators"],
                learning_rate=binary_cv.best_params_["classifier__learning_rate"],
                max_depth=binary_cv.best_params_["classifier__max_depth"],
                min_samples_split=binary_cv.best_params_[
                    "classifier__min_samples_split"
                ],
                subsample=binary_cv.best_params_["classifier__subsample"],
                random_state=42,
            ),
        ),
    ]
)
final_binary_classifier.fit(X_binary, y_binary)

# 2. Final regression model with RandomForestRegressor (trained only on spicy peppers)
final_regressor = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        (
            "regressor",
            ens.RandomForestRegressor(
                n_estimators=regression_cv.best_params_["regressor__n_estimators"],
                max_depth=regression_cv.best_params_["regressor__max_depth"],
                min_samples_split=regression_cv.best_params_[
                    "regressor__min_samples_split"
                ],
                min_samples_leaf=regression_cv.best_params_[
                    "regressor__min_samples_leaf"
                ],
                random_state=42,
            ),
        ),
    ]
)
final_regressor.fit(X_regression, y_regression)

# %% Apply two-step prediction to test data
# Step 1: Apply binary classifier to test data
binary_predictions = final_binary_classifier.predict(test_df)

# Step 2: For samples predicted as spicy we will apply the regression model
# For samples predicted as bell peppers (0), we set SHU to 0
final_predictions = np.zeros(len(test_df))
spicy_indices_test = binary_predictions == 1

if np.any(spicy_indices_test):  # Only predict if there are any spicy peppers
    # Get regression predictions for spicy samples
    regression_predictions = final_regressor.predict(test_df[spicy_indices_test])

    # Assign regression predictions to spicy samples
    final_predictions[spicy_indices_test] = regression_predictions

# Create DataFrame for submission
submission_df = pd.DataFrame(
    {
        # Convert back from log scale
        "Scoville Heat Units (SHU)": np.expm1(final_predictions)
    }
)
submission_df.index.name = "index"

# Save submission
base_dir = os.path.join("CA5", "results")
os.makedirs(base_dir, exist_ok=True)
filename = "gradientboost_randomforest_two_step_model_2_3cv.csv"
file_path = os.path.join(base_dir, filename)

submission_df.to_csv(file_path)
print(f"Saved GradientBoost-RandomForest two-step model submission to {file_path}")

# %% Visualize the prediction pipeline
# Optional: Evaluate the complete pipeline on the test split to see performance

# For binary classification
binary_acc = met.accuracy_score(y_test_binary, binary_cv.predict(X_test_binary))
print(f"Binary Classification Accuracy: {binary_acc:.3f}")

# %% [markdown]
# It was C that provided use with the best model. This makes sense since the data is skewed towards peppers that are sweet (0 SHU). So when we have seperated it so clearly by first using an ensemble model that can seperate between two binary values and then just use linear regressionIn order to get a better model we will need a larger dataset. We did earlier try with another model with several bins, but  there isnt enough data to make clear distinctions between the peppers.
