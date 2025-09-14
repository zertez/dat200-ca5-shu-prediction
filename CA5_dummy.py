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
    feature_selection as fsel,
)
from sklearn import (
    impute as imp,
)
from sklearn import (
    linear_model as lm,
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

# %%

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


# Load data
# 1. Load data
train_df = pd.read_csv("assets/train.csv")
test_df = pd.read_csv("assets/test.csv")


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
        df.loc[:, missing[missing].index] = imputer.fit_transform(df[missing[missing].index])

    if (missing := df[categorical_cols].isnull().any()).any():
        imputer = imp.SimpleImputer(strategy="most_frequent")
        df.loc[:, missing[missing].index] = imputer.fit_transform(df[missing[missing].index])

    return df


# Apply to train and test
train_df = impute_missing(train_df)
test_df = impute_missing(test_df)


# %% [markdown]
# We will give the long columns 12 and 13 shorter names

# %%
train_df = train_df.rename(columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"})
test_df = test_df.rename(columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"})

# Rename the storage temperature column
train_df = train_df.rename(columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"})
test_df = test_df.rename(columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"})

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
# Logarithmic transformation of SHU.

train_df["Scoville Heat Units (SHU)"] = np.log1p(train_df["Scoville Heat Units (SHU)"])

train_df["Seed Count"] = np.log1p(train_df["Seed Count"])
test_df["Seed Count"] = np.log1p(test_df["Seed Count"])

train_df["Vitamin C Content (mg)"] = np.log1p(train_df["Vitamin C Content (mg)"])
test_df["Vitamin C Content (mg)"] = np.log1p(test_df["Vitamin C Content (mg)"])

train_df["Weight (g)"] = np.log1p(train_df["Weight (g)"])
test_df["Weight (g)"] = np.log1p(test_df["Weight (g)"])

train_df["Moisture Content"] = np.log1p(train_df["Moisture Content"])
test_df["Moisture Content"] = np.log1p(test_df["Moisture Content"])


print(train_df.describe())


# %% [markdown]
# We need to labels based on SHU vales, we therefore found this image online and we follow the ranges like that.


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
# %% Removing color and harvesstime

train_df = train_df.drop(columns=["color", "Harvest Time"], axis=1)
test_df = test_df.drop(columns=["color", "Harvest Time"], axis=1)


# %% [markdown]


features = train_df.columns.drop(["spiciness_labels", "Scoville Heat Units (SHU)"])
classes = train_df["spiciness_labels"].unique()

fig, axes = plt.subplots(6, 2, figsize=(15, 30))
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

# Now that we have removed the categorical features color and harvest time.


# %%
# Define your features (excluding target and derived variables)
features = train_df.columns.drop(["spiciness_labels"])

# Create correlation matrix for the entire dataset
correlation_matrix = train_df[features].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Feature Engineering

# ## Feature Correlation Matrix


# %% [markdown]
# # Regression Models
# We'll implement a regression model to predict Scoville Heat Units (SHU).


# %%

X = train_df.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels"])
y = train_df["Scoville Heat Units (SHU)"]


# %% Splitting the data into train and test data with a 80/20 split
X_train, X_test, y_train, y_test = msel.train_test_split(X, y, test_size=0.2, random_state=42)
# %% [markdown]
# ## Ridge
# %% Ridge Pipeline
ridge_pipe = pipe.Pipeline([("scaler", prep.StandardScaler()), ("ridge", lm.Ridge(alpha=1, solver="svd"))])

# Capture warnings during model fitting

ridge_pipe.fit(X_train, y_train)

y_pred = ridge_pipe.predict(X_test)

mse = met.mean_squared_error(y_test, y_pred)
r2 = met.r2_score(y_test, y_pred)
print(f"Ridge - MSE: {mse:.4f}, R²: {r2:.4f}")
# %% [markdown]
# ## Linear Regression Pipe
# %% Linear Regression Pipeline
lm_pipe = pipe.Pipeline([("scaler", prep.StandardScaler()), ("linear_regression", lm.LinearRegression())])

# Capture warnings during model fitting

lm_pipe.fit(X_train, y_train)

y_pred = lm_pipe.predict(X_test)
mae = met.mean_absolute_error(y_test, y_pred)
r2 = met.r2_score(y_test, y_pred)
print(f"PCR - MAE: {mae:.4f}, R²: {r2:.4f}")
# %% [markdown]
# ## Gradient Boosting Pipe
# %% Gradient Boosting Pipeline
gbr_pipe = pipe.Pipeline([("scaler", prep.StandardScaler()), ("gbr", ens.GradientBoostingRegressor())])

# Define parameter grid for Grid Search
# Define wider parameter distributions for random search
param_distributions = {
    "gbr__n_estimators": np.arange(50, 300, 25),
    "gbr__learning_rate": np.logspace(-3, 0, 10),
    "gbr__max_depth": np.arange(3, 10),
    "gbr__min_samples_split": np.arange(2, 20, 2),
    "gbr__subsample": np.linspace(0.7, 1.0, 10),
}

# Create randomized search
random_search = msel.RandomizedSearchCV(
    estimator=gbr_pipe,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings to try
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    cv=3,
    random_state=42,
)

# Fit the grid search
print("Starting Grid Search...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("\nBest Parameters:")
print(random_search.best_params_)
print(f"Best Cross-Validation Score (Negative MAE): {random_search.best_score_:.4f}")

# Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = met.mean_absolute_error(y_test, y_pred)
r2 = met.r2_score(y_test, y_pred)
print(f"\nTest Set Performance - MAE: {mae:.4f}, R²: {r2:.4f}")


# %% [markdown]
# ## PCR Pipe
# %% PCR Pipeline
combined_pipe = pipe.Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        (
            "feature_selection",
            fsel.SequentialFeatureSelector(
                lm.LinearRegression(),
                n_features_to_select="auto",  # Or specify a number
                direction="backward",
                scoring="neg_mean_squared_error",
            ),
        ),
        ("linear_regression", lm.LinearRegression()),
    ]
)
combined_pipe.fit(X_train, y_train)

# %%
y_pred = combined_pipe.predict(X_test)
mae = met.mean_absolute_error(y_test, y_pred)
r2 = met.r2_score(y_test, y_pred)
print(f"PCR - MAE: {mae:.4f}, R²: {r2:.4f}")

# %% [markdown]
# ## Random Forest Regressor Pipe
# %% Random Forest Regressor

rf_reg = ens.RandomForestRegressor(n_jobs=-1, random_state=42)

rf_pipe = pipe.Pipeline([("rf_reg", rf_reg)])


# Parameter distributions for random search
param_grid_rf_reg = {
    "rf_reg__n_estimators": [100, 200, 300, 400, 500],
    "rf_reg__max_depth": [5, 10, 15, 20, 25, None],
    "rf_reg__min_samples_split": [2, 5, 10, 15, 20],
    "rf_reg__max_features": ["sqrt", "log2", None],
    "rf_reg__criterion": ["absolute_error", "squared_error", "friedman_mse"],
}


# We will use randomizedsearchCV since we are doing hyperparamter tuning
rf_reg_search = msel.RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_grid_rf_reg,
    n_iter=200,
    # Using StratifiedKFold
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)


rf_reg_search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % rf_reg_search.best_score_)
print(rf_reg_search.best_params_)


# Print best results
print("Best parameter (CV score=%0.3f):" % rf_reg_search.best_score_)
print(rf_reg_search.best_params_)

# %% Evaluate on test set
# Evaluate on test set
best_rf_model = rf_reg_search.best_estimator_
y_test_pred_rf = best_rf_model.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred_rf:.3f}")
print(best_rf_model)
# %% Detailed evaluation on test set from sample
# Get predictions on the sample test set
y_pred_rf = best_rf_model.predict(X_test)

print(f"MAE: {met.mean_absolute_error(y_test, y_pred_rf):.3f}")
print(f"RMSE: {np.sqrt(met.mean_squared_error(y_test, y_pred_rf)):.3f}")
print(f"R²: {met.r2_score(y_test, y_pred_rf):.3f}")

# %% Prepare full data
X_full = train_df.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels"])
y_full = train_df["Scoville Heat Units (SHU)"]

final_rf_reg = ens.RandomForestRegressor(
    n_estimators=rf_reg_search.best_params_["rf_reg__n_estimators"],
    max_features=rf_reg_search.best_params_["rf_reg__max_features"],
    max_depth=rf_reg_search.best_params_["rf_reg__max_depth"],
    criterion=rf_reg_search.best_params_["rf_reg__criterion"],
    min_samples_split=rf_reg_search.best_params_["rf_reg__min_samples_split"],
    random_state=42,
    n_jobs=-1,  # Use all available cores
)

# Create the final pipeline with scaling
final_rf_reg_pipeline = pipe.Pipeline([("rf_reg", final_rf_reg)])

# Train the pipeline on unscaled data - pipeline handles scaling internally
final_rf_reg_pipeline.fit(X_full, y_full)


# %% final pred


# %%
# %%
ypred = final_rf_reg_pipeline.predict(test_df)

# Create the final dataframe with numeric class predictions
ypred = pd.DataFrame(ypred, columns=["Scoville Heat Units (SHU)"])
ypred["Scoville Heat Units (SHU)"] = np.expm1(ypred["Scoville Heat Units (SHU)"])
ypred.index.name = "index"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA5", "results")
os.makedirs(base_dir, exist_ok=True)
filename = "rf_reg_model_3.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
ypred[["Scoville Heat Units (SHU)"]].to_csv(file_path)
print(f"Saved rf submission to {file_path}")


# %% [markdown]
# # Task C: Two-step Analysis with GradientBoosting and RandomForest

# First step: Create a binary classifier to separate bell peppers (SHU=0) from spicy peppers (SHU>0)
# We need to create a binary target variable first

# Convert log-transformed SHU back to original scale for creating binary labels
original_shu = np.expm1(train_df["Scoville Heat Units (SHU)"])


# %% Create binary labels (0 for bell peppers, 1 for spicy peppers)
train_df["is_spicy"] = (original_shu > 0).astype(int)

# Check distribution of bell vs spicy peppers
print("Distribution of pepper types:")
print(train_df["is_spicy"])

# %% Prepare data for binary classification
X_binary = train_df.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"])
y_binary = train_df["is_spicy"]

# Split data for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = msel.train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Create binary classifier pipeline with GradientBoostingClassifier
binary_pipe = pipe.Pipeline(
    [("scaler", prep.StandardScaler()), ("classifier", ens.GradientBoostingClassifier(random_state=42))]
)

# Define hyperparameter grid for tuning GradientBoostingClassifier
binary_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__learning_rate": [0.01, 0.1, 0.5],
    "classifier__max_depth": [3, 5, 7],
    "classifier__min_samples_split": [2, 5, 10],
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
X_regression = train_df[spicy_indices_train].drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"])
y_regression = train_df[spicy_indices_train]["Scoville Heat Units (SHU)"]

# %%

print(y_regression.describe())

# %% Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = msel.train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Create regression pipeline with RandomForestRegressor
regression_pipe = pipe.Pipeline(
    [("scaler", prep.StandardScaler()), ("regressor", ens.RandomForestRegressor(random_state=42))]
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
                min_samples_split=binary_cv.best_params_["classifier__min_samples_split"],
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
                min_samples_split=regression_cv.best_params_["regressor__min_samples_split"],
                min_samples_leaf=regression_cv.best_params_["regressor__min_samples_leaf"],
                random_state=42,
            ),
        ),
    ]
)
final_regressor.fit(X_regression, y_regression)

# %% Apply two-step prediction to test data

# Step 1: Apply binary classifier to test data
binary_predictions = final_binary_classifier.predict(test_df)

# Step 2: For samples predicted as spicy, apply regression model
# For samples predicted as bell peppers (0), set SHU to 0
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
