import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CA5

    Group 37

    Group members:
    * Jannicke Ådalen
    * Marcus Dalaker Figenschou
    * Rikke Sellevold Vegstein
    """
    )
    return


@app.cell
def _():
    # Standard library imports
    import os

    import marimo as mo
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
    from sklearn import (
        svm as svm,
    )

    return mo, os, plt, np, pd, sns, ens, imp, lm, met, msel, pipe, prep, svm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Heisenbug
    There is right now apparently a "Heisenbug" which only affects macs with the M4 chips that one in the group members uses. We therefore have to set numpy to ignore all warnings to prevent RuntimeWarning messages from being displayed.
    https://github.com/numpy/numpy/issues/28687, due to this error we are quite limited to selecting our models.
    """
    )
    return


@app.cell
def _(np):
    np.seterr(all="ignore")
    return


@app.cell
def _(os, pd, plt, sns):
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
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data inspection and cleaning""")
    return


@app.cell
def _(test_df, train_df):
    print("---TRAIN DATA---")
    train_df.info()

    print("---TEST DATA---")
    test_df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Both the test and training data er missing values, we will therefore do imputation and add values to the missing columns based on their mean values."""
    )
    return


@app.cell
def _(imp, test_df, train_df):
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

    train_df_1 = impute_missing(train_df)
    test_df_1 = impute_missing(test_df)
    return test_df_1, train_df_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Checking""")
    return


@app.cell
def _(test_df_1, train_df_1):
    print("---TRAIN DATA---")
    train_df_1.info()
    print("---TEST DATA---")
    test_df_1.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The imputation has worked.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will shorten the names of column 12 and 13 for better handling""")
    return


@app.cell
def _(test_df_1, train_df_1):
    train_df_2 = train_df_1.rename(columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"})
    test_df_2 = test_df_1.rename(columns={"Average Daily Temperature During Growth (celcius)": "Avg Growth Temp (C)"})
    train_df_2 = train_df_2.rename(columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"})
    test_df_2 = test_df_2.rename(columns={"Average Temperature During Storage (celcius)": "Avg Storage Temp (C)"})
    print("---TRAIN DATA---")
    train_df_2.info()
    print("---TEST DATA---")
    test_df_2.info()
    return test_df_2, train_df_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Statistical inpsection""")
    return


@app.cell
def _(train_df_2):
    print(train_df_2.describe())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We will drop storage temp, color and harvest time since these are object types and it will be complicated to interpret these"""
    )
    return


@app.cell
def _(test_df_2, train_df_2):
    train_df_3 = train_df_2.drop(columns=["Avg Storage Temp (C)", "color", "Harvest Time"], axis=1)
    test_df_3 = test_df_2.drop(columns=["Avg Storage Temp (C)", "color", "Harvest Time"], axis=1)
    return test_df_3, train_df_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Log transformation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We are dealing with regression now which is sensitive to large value ranges, we can see that we have multiple features that have large value ranges such as: seed count, vitamin c, weight and moisture. Also SHU has a large value range. We will thereofre use np.log1p which applies the log(1+x). This is to prevent it from taking log of 0 which is not possible. We must also after we have made our predictions revert back to the normal SHU scale."""
    )
    return


@app.cell
def _(np, test_df_3, train_df_3):
    features_to_log = ["Weight (g)", "Seed Count", "Moisture Content", "Vitamin C Content (mg)"]
    for feature in features_to_log:
        train_df_3[feature] = np.log1p(train_df_3[feature])
        test_df_3[feature] = np.log1p(test_df_3[feature])
    train_df_3["Scoville Heat Units (SHU)"] = np.log1p(train_df_3["Scoville Heat Units (SHU)"])
    print(train_df_3.describe())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Labelling""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will add labels to SHU based on spicyness here for easier plotting and interpretation""")
    return


@app.cell
def _(np, pd, train_df_3):
    bins = [0, np.log1p(1), np.log1p(5001), np.log1p(15001), np.log1p(100001), np.log1p(350001), float("inf")]
    labels = ["Sweet", "Mild", "Medium", "Medium-Hot", "Hot", "Superhot"]
    train_df_3["spiciness_labels"] = pd.cut(
        train_df_3["Scoville Heat Units (SHU)"], bins=bins, labels=labels, include_lowest=True
    )
    print(train_df_3[["Scoville Heat Units (SHU)", "spiciness_labels"]].head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Violinplot""")
    return


@app.cell
def _(plt, sns, train_df_3):
    features = train_df_3.columns.drop(["spiciness_labels", "Scoville Heat Units (SHU)"])
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, feature_1 in enumerate(features):
        sns.violinplot(
            data=train_df_3,
            x="spiciness_labels",
            y=feature_1,
            hue="spiciness_labels",
            palette="Set1",
            ax=axes[i],
            alpha=0.4,
            orient="v",
        )
        axes[i].set_title(f"Violinplot of {feature_1}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", labelsize=14)
        axes[i].patch.set_edgecolor("black")
        axes[i].patch.set_linewidth(1)
    fig.delaxes(axes[11])
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Most of the features have a unimodal distribtuion, so we can safely use standardscaler here. Looking at mild pepeprs, this is the feature that is themost skewed , this is most likely just due to the range we have chosen. We can see here that there are very few features that shows clear separation between the classes, such as pericarp thickness, capsaicin content, sugar content and seed count. We will therefore expect that these are our most important features."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Correlation Matrix""")
    return


@app.cell
def _(plt, sns, train_df_3):
    features_1 = train_df_3.columns.drop(["spiciness_labels"])
    correlation_matrix = train_df_3[features_1].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can see here that most of the features exccept of width and avg growth temp have importance towards describing Scoville heat units. The most important features are pericarp thickness, seed count, capsaicin content and sugar content. This makes sense, the less seeds the more spicy the pepper is, more sugar equals spicier pepper and more capsaicin the spicier the pepper. Our corr matrix confirms the most important features, that we guessed from the violinplot."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Regression Models
    We'll implement a regression model to predict Scoville Heat Units (SHU).
    """
    )
    return


@app.cell
def _(train_df_3):
    X = train_df_3.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels"])
    y = train_df_3["Scoville Heat Units (SHU)"]
    return X, y


@app.cell
def _(X, msel, y):
    X_train, X_test, y_train, y_test = msel.train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ridge""")
    return


@app.cell
def _(X_test, X_train, lm, met, pipe, prep, y_test, y_train):
    ridge_pipe = pipe.Pipeline(
        [
            ("scaler", prep.StandardScaler()),
            # This doesnt work with l1 and l2 because of the Heisenbug
            ("ridge", lm.Ridge(alpha=1, solver="auto")),
        ]
    )

    # Capture warnings during model fitting

    ridge_pipe.fit(X_train, y_train)

    y_pred_ridge = ridge_pipe.predict(X_test)

    mae = met.mean_absolute_error(y_test, y_pred_ridge)
    r2 = met.r2_score(y_test, y_pred_ridge)
    print(f"Ridge - MAE: {mae:.4f}, R²: {r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Linear Regression Pipe""")
    return


@app.cell
def _(X_test, X_train, lm, met, pipe, prep, y_test, y_train):
    lm_pipe = pipe.Pipeline([("scaler", prep.StandardScaler()), ("linear_regression", lm.LinearRegression())])
    lm_pipe.fit(X_train, y_train)
    y_pred_lm = lm_pipe.predict(X_test)
    mae_1 = met.mean_absolute_error(y_test, y_pred_lm)
    r2_1 = met.r2_score(y_test, y_pred_lm)
    print(f"PCR - MAE: {mae_1:.4f}, R²: {r2_1:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Gradient Boosting Pipe""")
    return


@app.cell
def _(X_test, X_train, ens, met, msel, np, pipe, prep, y_test, y_train):
    gbr_pipe = pipe.Pipeline([("scaler", prep.StandardScaler()), ("gbr", ens.GradientBoostingRegressor())])
    param_grid = {
        "gbr__n_estimators": np.array([50, 100, 150, 200, 250]),
        "gbr__learning_rate": np.logspace(-3, 0, 4),
        "gbr__max_depth": np.array([3, 6, 9]),
        "gbr__min_samples_split": np.array([2, 8, 16]),
        "gbr__subsample": np.array([0.7, 0.85, 1.0]),
    }
    random_search = msel.GridSearchCV(
        estimator=gbr_pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        cv=3,
        return_train_score=True,
    )
    print("Starting Grid Search...")
    random_search.fit(X_train, y_train)
    print("\nBest Parameters:")
    print(random_search.best_params_)
    print(f"Best Cross-Validation Score (Negative MAE): {random_search.best_score_:.4f}")
    best_model = random_search.best_estimator_
    y_pred_gb = best_model.predict(X_test)
    mae_2 = met.mean_absolute_error(y_test, y_pred_gb)
    r2_2 = met.r2_score(y_test, y_pred_gb)
    print(f"\nTest Set Performance - MAE: {mae_2:.4f}, R²: {r2_2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Random Forest Regressor Pipe""")
    return


@app.cell
def _(X_train, ens, msel, pipe, y_train):
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

    # Using gridsearch with cv set to 3
    rf_reg_search = msel.GridSearchCV(
        rf_pipe,
        param_grid=param_grid_rf_reg,
        cv=3,
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
    return (rf_reg_search,)


@app.cell
def _(X_test, rf_reg_search, y_test):
    # Evaluate on test set
    best_rf_model = rf_reg_search.best_estimator_
    y_test_pred_rf = best_rf_model.score(X_test, y_test)
    print(f"Test set score with best model: {y_test_pred_rf:.3f}")
    print(best_rf_model)
    return (best_rf_model,)


@app.cell
def _(X_test, best_rf_model, met, y_test):
    # Get predictions on the sample test set
    y_pred_rf = best_rf_model.predict(X_test)

    print(f"MAE: {met.mean_absolute_error(y_test, y_pred_rf):.3f}")
    print(f"R²: {met.r2_score(y_test, y_pred_rf):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The best model is the random forest regression model, which has the lowest MAE of 2.145 and the highest R^2 of 0.622."""
    )
    return


@app.cell
def _(ens, pipe, rf_reg_search, train_df_3):
    X_full = train_df_3.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels"])
    y_full = train_df_3["Scoville Heat Units (SHU)"]
    final_rf_reg = ens.RandomForestRegressor(
        n_estimators=rf_reg_search.best_params_["rf_reg__n_estimators"],
        max_features=rf_reg_search.best_params_["rf_reg__max_features"],
        max_depth=rf_reg_search.best_params_["rf_reg__max_depth"],
        criterion=rf_reg_search.best_params_["rf_reg__criterion"],
        min_samples_split=rf_reg_search.best_params_["rf_reg__min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )
    final_rf_reg_pipeline = pipe.Pipeline([("rf_reg", final_rf_reg)])
    final_rf_reg_pipeline.fit(X_full, y_full)
    return (final_rf_reg_pipeline,)


@app.cell
def _(final_rf_reg_pipeline, np, os, pd, test_df_3):
    ypred = final_rf_reg_pipeline.predict(test_df_3)
    ypred = pd.DataFrame(ypred, columns=["Scoville Heat Units (SHU)"])
    ypred["Scoville Heat Units (SHU)"] = np.expm1(ypred["Scoville Heat Units (SHU)"])
    ypred.index.name = "index"
    base_dir = os.path.join("CA5", "results")
    os.makedirs(base_dir, exist_ok=True)
    filename = "rf_reg_model_3.csv"
    file_path = os.path.join(base_dir, filename)
    ypred[["Scoville Heat Units (SHU)"]].to_csv(file_path)
    print(f"Saved rf submission to {file_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Task C: Two-step Analysis with GradientBoosting and RandomForest

    First step: Create a binary classifier to separate bell peppers (SHU=0) from spicy peppers (SHU>0)
    We need to create a binary target variable first. Before that we must convert the log-transformed SHU back to original scale for creating the binary labels. We will make another column for this.
    """
    )
    return


@app.cell
def _(np, train_df_3):
    original_shu = np.expm1(train_df_3["Scoville Heat Units (SHU)"])
    return (original_shu,)


@app.cell
def _(original_shu, train_df_3):
    train_df_3["is_spicy"] = (original_shu > 0).astype(int)
    print("Distribution of pepper types:")
    print(train_df_3["is_spicy"])
    return


@app.cell
def _(ens, met, msel, pipe, prep, train_df_3):
    X_binary = train_df_3.drop(columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"])
    y_binary = train_df_3["is_spicy"]
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = msel.train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    binary_pipe = pipe.Pipeline(
        [("scaler", prep.StandardScaler()), ("classifier", ens.GradientBoostingClassifier(random_state=42))]
    )
    binary_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.5],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__subsample": [0.8, 1.0],
    }
    binary_cv = msel.GridSearchCV(
        binary_pipe, param_grid=binary_grid, cv=3, n_jobs=-1, scoring="f1", return_train_score=True, verbose=1
    )
    binary_cv.fit(X_train_binary, y_train_binary)
    print("Best parameters for binary classifier:", binary_cv.best_params_)
    print("Best binary classification score:", binary_cv.best_score_)
    y_pred_binary = binary_cv.predict(X_test_binary)
    print("\nBinary Classification Report:")
    print(met.classification_report(y_test_binary, y_pred_binary))
    return X_binary, X_test_binary, binary_cv, y_binary, y_test_binary


@app.cell
def _(train_df_3):
    spicy_indices_train = train_df_3["is_spicy"] == 1
    X_regression = train_df_3[spicy_indices_train].drop(
        columns=["Scoville Heat Units (SHU)", "spiciness_labels", "is_spicy"]
    )
    y_regression = train_df_3[spicy_indices_train]["Scoville Heat Units (SHU)"]
    return X_regression, y_regression


@app.cell
def _(y_regression):
    print(y_regression.describe())
    return


@app.cell
def _(X_regression, ens, met, msel, pipe, prep, y_regression):
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
    return (regression_cv,)


@app.cell
def _(
    X_binary,
    X_regression,
    binary_cv,
    ens,
    pipe,
    prep,
    regression_cv,
    y_binary,
    y_regression,
):
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
    return final_binary_classifier, final_regressor


@app.cell
def _(final_binary_classifier, final_regressor, np, os, pd, test_df_3):
    binary_predictions = final_binary_classifier.predict(test_df_3)
    final_predictions = np.zeros(len(test_df_3))
    spicy_indices_test = binary_predictions == 1
    if np.any(spicy_indices_test):
        regression_predictions = final_regressor.predict(test_df_3[spicy_indices_test])
        final_predictions[spicy_indices_test] = regression_predictions
    submission_df = pd.DataFrame({"Scoville Heat Units (SHU)": np.expm1(final_predictions)})
    submission_df.index.name = "index"
    base_dir_1 = os.path.join("CA5", "results")
    os.makedirs(base_dir_1, exist_ok=True)
    filename_1 = "gradientboost_randomforest_two_step_model_2_3cv.csv"
    file_path_1 = os.path.join(base_dir_1, filename_1)
    submission_df.to_csv(file_path_1)
    print(f"Saved GradientBoost-RandomForest two-step model submission to {file_path_1}")
    return


@app.cell
def _(X_test_binary, binary_cv, met, y_test_binary):
    # Optional: Evaluate the complete pipeline on the test split to see performance

    # For binary classification
    binary_acc = met.accuracy_score(y_test_binary, binary_cv.predict(X_test_binary))
    print(f"Binary Classification Accuracy: {binary_acc:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""It was C that provided use with the best model. This makes sense since the data is skewed towards peppers that are sweet (0 SHU). So when we have seperated it so clearly by first using an ensemble model that can seperate between two binary values and then just use linear regressionIn order to get a better model we will need a larger dataset. We did earlier try with another model with several bins, but  there isnt enough data to make clear distinctions between the peppers."""
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
