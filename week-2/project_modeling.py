from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from xgboost import XGBRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Hyperparameter Distributions
# -----------------------------------------------------------

linreg_params = {
    "model__fit_intercept": [True, False]
}


linreg_select_params = {
    "select__k": randint(30, 150),
    "model__fit_intercept": [True, False]
}

rf_params = {
    "model__n_estimators": randint(200, 600),
    "model__criterion": ["squared_error", "friedman_mse"],
    "model__max_depth": randint(6, 20),
    "model__min_samples_split": randint(5, 30),
    "model__min_samples_leaf": randint(2, 15),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap": [True]
}


rf_select_params = {
    "select__k": randint(30, 150),
    "model__n_estimators": randint(200, 600),
    "model__criterion": ["squared_error", "friedman_mse"],
    "model__max_depth": randint(6, 20),
    "model__min_samples_split": randint(5, 30),
    "model__min_samples_leaf": randint(2, 15),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "model__bootstrap": [True]
}


dt_params = {
    "model__criterion": ["squared_error", "friedman_mse"],
    "model__max_depth": randint(3, 16),
    "model__min_samples_split": randint(5, 30),
    "model__min_samples_leaf": randint(3, 20),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
}

gb_params = {
    "model__n_estimators": randint(100, 500),
    "model__learning_rate": loguniform(0.01, 0.15),
    "model__max_depth": randint(2, 5),
    "model__min_samples_split": randint(5, 30),
    "model__min_samples_leaf": randint(3, 20),
    "model__subsample": uniform(0.7, 0.3),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
}


gb_select_params = {
    "select__k": randint(30, 150),
    "model__n_estimators": randint(100, 500),
    "model__learning_rate": loguniform(0.01, 0.15),
    "model__max_depth": randint(2, 5),
    "model__min_samples_split": randint(5, 30),
    "model__min_samples_leaf": randint(3, 20),
    "model__subsample": uniform(0.7, 0.3),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
}


xgb_params = {
    "model__n_estimators": randint(200, 800),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_depth": randint(2, 8),
    "model__min_child_weight": randint(1, 10),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__gamma": uniform(0, 5),
    "model__reg_alpha": loguniform(1e-4, 1),
    "model__reg_lambda": loguniform(0.1, 10)
}


xgb_select_params = {
    "select__k": randint(30, 150),
    "model__n_estimators": randint(200, 800),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_depth": randint(2, 8),
    "model__min_child_weight": randint(1, 10),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__gamma": uniform(0, 5),
    "model__reg_alpha": loguniform(1e-4, 1),
    "model__reg_lambda": loguniform(0.1, 10)
}


# -----------------------------------------------------------
# models hyperparameter distribution dict
# -----------------------------------------------------------
cv = KFold( 
    n_splits=5,
    shuffle=True,
    random_state=42
)



param_distributions = {
    "Linear Regression": linreg_params,
    "Linear Regression + SelectKBest": linreg_select_params,
    "Decision Tree": dt_params,
    "Random Forest": rf_params,
    "Random Forest + SelectKBest": rf_select_params,
    "Gradient Boosting": gb_params,
    "Gradient Boosting + SelectKBest": gb_select_params,
    "XGBoost": xgb_params,
    "XGBoost + SelectKBest": xgb_select_params
}

n_iters = {
    "Linear Regression": 2,
    "Linear Regression + SelectKBest": 30,
    "Decision Tree": 30,
    "Random Forest": 30,
    "Random Forest + SelectKBest": 30,
    "Gradient Boosting": 30,
    "Gradient Boosting + SelectKBest": 30,
    "XGBoost": 30,
    "XGBoost + SelectKBest": 30
}



# ============================================================
# Func to find best models through RandomizedSearchCV
# ============================================================

def search_regression_model(
    model,
    param_distributions,
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    cv, 
    n_iter=30
):
    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        refit="rmse",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    train_preds = best_model.predict(X_train)
    valid_preds = best_model.predict(X_valid)

    train_mae = mean_absolute_error(y_train, train_preds)
    valid_mae = mean_absolute_error(y_valid, valid_preds)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))

    train_r2 = r2_score(y_train, train_preds)
    valid_r2 = r2_score(y_valid, valid_preds)

    best_cv_mae = -search.cv_results_["mean_test_mae"][search.best_index_]
    best_cv_rmse = -search.cv_results_["mean_test_rmse"][search.best_index_]
    best_cv_r2 = search.cv_results_["mean_test_r2"][search.best_index_]

    print("\nBest CV RMSE:", best_cv_rmse)
    print("Best CV MAE:", best_cv_mae)
    print("Best CV R2:", best_cv_r2)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nValidation MAE:", valid_mae)
    print("Validation RMSE:", valid_rmse)
    print("Validation R2:", valid_r2)

    return {
        "model_name": model_name,
        "best_cv_mae": best_cv_mae,
        "best_cv_rmse": best_cv_rmse,
        "best_cv_r2": best_cv_r2,
        "train_mae": train_mae,
        "valid_mae": valid_mae,
        "train_rmse": train_rmse,
        "valid_rmse": valid_rmse,
        "train_r2": train_r2,
        "valid_r2": valid_r2,
        "best_params": search.best_params_,
        "best_model": best_model,
        "search_object": search,
        "valid_preds": valid_preds
    }

def run_regression_searches(
    models,
    param_distributions,
    X_train,
    y_train,
    X_valid,
    y_valid,
    cv,
    n_iters=None
):

    if n_iters is None:
        n_iters = {}

    results = {}

    for model_name, model in models.items():
        print(f"\nRunning search for: {model_name}")

        if model_name not in param_distributions:
            raise ValueError(f"No parameter distribution found for: {model_name}")

        results[model_name] = search_regression_model(
            model=model,
            param_distributions=param_distributions[model_name],
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            cv=cv,
            n_iter=n_iters.get(model_name, 30)
        )

    return results

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from xgboost import XGBRegressor


def build_regression_models(preprocessor, k=100, random_state=42):
    models = {
        "Linear Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),

        "Linear Regression + SelectKBest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
            ("model", LinearRegression())
        ]),

        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=random_state))
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state))
        ]),

        "Random Forest + SelectKBest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
            ("model", RandomForestRegressor(random_state=random_state))
        ]),

        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=random_state))
        ]),

        "Gradient Boosting + SelectKBest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
            ("model", GradientBoostingRegressor(random_state=random_state))
        ]),

        "XGBoost": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(random_state=random_state))
        ]),

        "XGBoost + SelectKBest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
            ("model", XGBRegressor(random_state=random_state))
        ])
    }

    return models


def make_model_summary(results):
    if isinstance(results, dict):
        results = list(results.values())

    summary_df = pd.DataFrame(results)[[
        "model_name",
        "best_cv_mae",
        "best_cv_rmse",
        "best_cv_r2",
        "train_mae",
        "valid_mae",
        "train_rmse",
        "valid_rmse",
        "train_r2",
        "valid_r2",
        "best_params"
    ]]

    summary_df = summary_df.sort_values(
        by="valid_rmse",
        ascending=True
    )

    return summary_df


### VISUALISATIONS


def plot_training_history(history):
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)

    plt.figure(figsize=(8, 5))

    sns.lineplot(data=history_df, x="epoch", y="loss", label="Train Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_loss", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss During Training")
    plt.grid(True)
    plt.show()

    return history_df

def clean_transformed_feature_names(feature_series):
    return (
        feature_series
        .str.replace("full_cat_ordinal__", "", regex=False)
        .str.replace("full_cat_nominal__", "", regex=False)
        .str.replace("structural_num__", "", regex=False)
        .str.replace("structural_cat_ordinal__", "", regex=False)
        .str.replace("structural_cat_nominal__", "", regex=False)
        .str.replace("full_num__", "", regex=False)
        .str.replace("ordinary_num__", "", regex=False)
        .str.replace("ordinary_cat__", "", regex=False)
        .str.replace("remainder__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )


def plot_selected_model_feature_importance(
    model,
    top_n=15,
    title="Top Feature Importances"
):
    all_feature_names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    if "select" in model.named_steps:
        selected_mask = model.named_steps["select"].get_support()
        feature_names = all_feature_names[selected_mask]
    else:
        feature_names = all_feature_names

    importances = model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df["feature_clean"] = clean_transformed_feature_names(
        importance_df["feature"]
    )

    top_features = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_features["feature_clean"],
        top_features["importance"]
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Feature Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(axis="x", alpha=0.25)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.show()

    return importance_df


import matplotlib.ticker as ticker
def plot_actual_vs_predicted(y_true, y_pred, target_name="SalePrice", title="Actual vs Predicted"):
    plt.figure(figsize=(7, 7))

    plt.scatter(y_true, y_pred, alpha=0.5)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect prediction"
    )

    def k_formatter(x, pos):
        return f"{int(x / 1000)}k"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))

    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals vs Predicted Value"):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")

    def k_formatter(x, pos):
        return f"{int(x / 1000)}k"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))

    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return residuals

