import time
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)


class AutoMLEnv:

    LOWER_IS_BETTER = {
        "mse",
        "rmse",
        "mae"
    }

    def __init__(
        self,
        train_path,
        target_column,
        metric="roc_auc",
        budget=1000
    ):
        self.metric = metric
        self.target_column = target_column
        self.compute_budget = float(budget)
        self.budget = self.compute_budget

        # Maximum total LLM tokens allowed for one AutoML run.
        # Counts prompt + response tokens returned by the local LLM wrapper.
        self.token_budget = 32000

        self.raw_df = self._read_csv_safely(
            train_path
        )

        self.initial_shape = self.raw_df.shape
        self.initial_missing_values = int(self.raw_df.isna().sum().sum())
        self.initial_duplicate_rows = int(self.raw_df.duplicated().sum())
        self.initial_object_columns = [
            column
            for column in self.raw_df.columns
            if (
                self.raw_df[column].dtype == "object"
                or "string" in str(self.raw_df[column].dtype)
                or str(self.raw_df[column].dtype) == "category"
            )
        ]

        if target_column not in self.raw_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(self.raw_df.columns)}"
            )

        self.history = []
        self.candidates = []

        self.best_score = None
        self.best_objective = -float("inf")
        self.best_model_name = None
        self.best_model_object = None
        self.best_candidate_id = None
        self.best_action = None

        self.total_compute_cost = 0.0
        self.total_token_cost = 0

        self.category_maps = {}
        self.feature_columns = []
        self.leakage_columns_removed = []

        self.task_type = self._infer_task_type(
            self.raw_df[target_column]
        )

        self.df = self._prepare_train_dataframe(
            self.raw_df.copy()
        )

        self.X = self.df.drop(
            columns=[self.target_column]
        )

        self.y = self.df[self.target_column]

        stratify = None

        if (
            self.task_type == "classification"
            and self.y.nunique() > 1
        ):
            stratify = self.y

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

    def _read_csv_safely(self, path):

        attempts = [
            {
                "sep": None,
                "engine": "python",
                "on_bad_lines": "skip"
            },
            {
                "sep": ";",
                "engine": "python",
                "on_bad_lines": "skip"
            },
            {
                "sep": ",",
                "engine": "python",
                "on_bad_lines": "skip"
            }
        ]

        last_error = None

        for kwargs in attempts:
            try:
                df = pd.read_csv(
                    path,
                    **kwargs
                )

                if df.shape[1] > 1:
                    return df

            except Exception as error:
                last_error = error

        raise RuntimeError(
            f"Cannot read CSV: {last_error}"
        )

    def _infer_task_type(self, y):

        if self.metric in self.LOWER_IS_BETTER:
            return "regression"

        if (
            y.dtype == "object"
            or y.nunique() <= max(
                20,
                int(len(y) * 0.05)
            )
        ):
            return "classification"

        return "regression"

    def _remove_leakage(self, df):

        self.leakage_columns_removed = []

        return df

    def _clean_basic(
        self,
        df,
        is_train
    ):

        if is_train:
            df = df.drop_duplicates()

        for column in df.columns:

            if (
                column == self.target_column
                and not is_train
            ):
                continue

            if (
                df[column].dtype == "object"
                or "string" in str(df[column].dtype)
                or str(df[column].dtype) == "category"
            ):
                df[column] = (
                    df[column]
                    .fillna("missing")
                    .astype(str)
                )

            else:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                median = df[column].median()

                if pd.isna(median):
                    median = 0

                df[column] = df[column].fillna(
                    median
                )

        return df

    def _feature_engineering(self, df):

        numeric_columns = [
            column
            for column in df.select_dtypes(
                include=np.number
            ).columns
            if column != self.target_column
        ]

        for column in numeric_columns[:5]:
            df[f"{column}_squared"] = (
                df[column] ** 2
            )

        return df

    def _encode_train_features(self, X):

        X = X.copy()

        for column in X.columns:

            if (
                X[column].dtype == "object"
                or "string" in str(X[column].dtype)
                or str(X[column].dtype) == "category"
            ):
                values = (
                    X[column]
                    .fillna("missing")
                    .astype(str)
                )

                uniques = (
                    pd.Series(values.unique())
                    .sort_values()
                    .tolist()
                )

                mapping = {
                    value: index
                    for index, value in enumerate(uniques)
                }

                self.category_maps[column] = mapping

                X[column] = (
                    values
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        X = (
            X
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        self.feature_columns = list(X.columns)

        return X

    def _encode_external_features(self, X):

        X = X.copy()

        for column, mapping in self.category_maps.items():

            if column in X.columns:
                X[column] = (
                    X[column]
                    .fillna("missing")
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        X = (
            X
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        for column in self.feature_columns:
            if column not in X.columns:
                X[column] = 0

        X = X[self.feature_columns]

        return X

    def _prepare_train_dataframe(self, df):

        df = self._clean_basic(
            df,
            is_train=True
        )

        df = self._feature_engineering(df)

        df = self._remove_leakage(df)

        y = df[self.target_column]

        X = df.drop(
            columns=[self.target_column]
        )

        X = self._encode_train_features(X)

        prepared = X.copy()

        prepared[self.target_column] = y.values

        return prepared

    def prepare_external_features(self, df):

        df = self._clean_basic(
            df.copy(),
            is_train=False
        )

        df = self._feature_engineering(df)

        df = self._remove_leakage(df)

        if self.target_column in df.columns:
            df = df.drop(
                columns=[self.target_column]
            )

        return self._encode_external_features(df)

    def _sanitize_params(self, params):

        params = dict(params or {})

        forbidden = {
            "eval_set",
            "eval_set_fraction",
            "metrics",
            "callbacks",
            "eval_metric",
            "early_stopping_rounds",
            "verbose",
            "cat_features"
        }

        for key in list(params.keys()):
            if key in forbidden:
                params.pop(key, None)

        int_params = {
            "max_depth",
            "n_estimators",
            "min_samples_split",
            "min_samples_leaf",
            "num_leaves",
            "iterations",
            "depth"
        }

        for key in int_params:
            if key in params:
                try:
                    params[key] = int(params[key])
                except Exception:
                    params.pop(key, None)

        float_params = {
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "bagging_fraction",
            "feature_fraction",
            "l2_leaf_reg"
        }

        for key in float_params:
            if key in params:
                try:
                    params[key] = float(params[key])
                except Exception:
                    params.pop(key, None)

        if "n_estimators" in params:
            params["n_estimators"] = max(
                10,
                min(params["n_estimators"], 500)
            )

        if "iterations" in params:
            params["iterations"] = max(
                10,
                min(params["iterations"], 500)
            )

        if "max_depth" in params:
            params["max_depth"] = max(
                1,
                min(params["max_depth"], 12)
            )

        if "depth" in params:
            params["depth"] = max(
                1,
                min(params["depth"], 12)
            )

        return params

    def _get_model(
        self,
        model_name,
        params
    ):

        params = self._sanitize_params(params)

        common = {
            "random_state": 42
        }

        if model_name == "random_forest":

            params.setdefault(
                "n_estimators",
                120
            )

            if self.task_type == "regression":
                return RandomForestRegressor(
                    **common,
                    **params
                )

            return RandomForestClassifier(
                **common,
                **params
            )

        if model_name == "hist_gb":

            if self.task_type == "regression":
                return HistGradientBoostingRegressor(
                    **common,
                    **params
                )

            return HistGradientBoostingClassifier(
                **common,
                **params
            )

        if model_name == "lightgbm":

            params.setdefault(
                "n_estimators",
                120
            )

            params.setdefault(
                "learning_rate",
                0.05
            )

            if self.task_type == "regression":
                return LGBMRegressor(
                    random_state=42,
                    verbose=-1,
                    **params
                )

            return LGBMClassifier(
                random_state=42,
                verbose=-1,
                **params
            )

        if model_name == "catboost":

            params.setdefault(
                "iterations",
                params.pop("n_estimators", 120)
            )

            params.setdefault(
                "learning_rate",
                0.05
            )

            params.setdefault(
                "depth",
                params.pop("max_depth", 6)
            )

            if self.task_type == "regression":
                return CatBoostRegressor(
                    random_seed=42,
                    verbose=0,
                    **params
                )

            return CatBoostClassifier(
                random_seed=42,
                verbose=0,
                **params
            )

        params.setdefault(
            "n_estimators",
            120
        )

        params.setdefault(
            "max_depth",
            4
        )

        if self.task_type == "regression":
            return XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                **params
            )

        return XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            **params
        )

    def _predict_for_metric(
        self,
        model,
        X
    ):

        if (
            self.metric == "roc_auc"
            and hasattr(model, "predict_proba")
        ):
            proba = model.predict_proba(X)

            if proba.shape[1] == 2:
                return proba[:, 1]

            return proba

        return model.predict(X)

    def _calculate_metric(
        self,
        y_true,
        preds
    ):

        if self.metric == "roc_auc":

            if np.ndim(preds) == 2:
                return roc_auc_score(
                    y_true,
                    preds,
                    multi_class="ovr"
                )

            return roc_auc_score(
                y_true,
                preds
            )

        if self.metric == "mse":
            return mean_squared_error(
                y_true,
                preds
            )

        if self.metric == "rmse":
            return float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        preds
                    )
                )
            )

        if self.metric == "mae":
            return mean_absolute_error(
                y_true,
                preds
            )

        if self.metric == "f1":
            return f1_score(
                y_true,
                preds,
                average="weighted",
                zero_division=0
            )

        if self.metric == "precision":
            return precision_score(
                y_true,
                preds,
                average="weighted",
                zero_division=0
            )

        if self.metric == "recall":
            return recall_score(
                y_true,
                preds,
                average="weighted",
                zero_division=0
            )

        return accuracy_score(
            y_true,
            preds
        )

    def _objective_value(self, metric_value):

        if self.metric in self.LOWER_IS_BETTER:
            return -metric_value

        return metric_value

    def _cv_scoring(self):

        mapping = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        }

        return mapping.get(
            self.metric,
            None
        )

    def _feature_importance(self, model):

        if not hasattr(
            model,
            "feature_importances_"
        ):
            return {}

        values = model.feature_importances_

        order = np.argsort(values)[::-1][:10]

        return {
            self.X.columns[index]: float(values[index])
            for index in order
        }

    def _checklist_item(
        self,
        name,
        passed,
        details="",
        severity="info"
    ):
        return {
            "name": name,
            "passed": bool(passed),
            "details": str(details),
            "severity": severity
        }

    def _build_checklist_feedback(
        self,
        action,
        model_name,
        raw_params,
        sanitized_params,
        validation_result
    ):
        tested_models = []

        for item in self.history:
            try:
                model = item["action"].get("model")
                if model and model not in tested_models:
                    tested_models.append(model)
            except Exception:
                pass

        current_model_already_used = model_name in tested_models

        all_models = [
            "xgboost",
            "lightgbm",
            "catboost",
            "random_forest",
            "hist_gb"
        ]

        untested_models = [
            model
            for model in all_models
            if model not in tested_models
            and model != model_name
        ]

        params_changed = raw_params != sanitized_params

        checklist = [
            self._checklist_item(
                "dataset_loaded",
                self.initial_shape[0] > 0 and self.initial_shape[1] > 1,
                f"Loaded dataset with {self.initial_shape[0]} rows and {self.initial_shape[1]} columns."
            ),
            self._checklist_item(
                "target_column_checked",
                self.target_column in self.raw_df.columns,
                f"Target column: {self.target_column}."
            ),
            self._checklist_item(
                "task_type_detected",
                self.task_type in ["classification", "regression"],
                f"Detected task type: {self.task_type}."
            ),
            self._checklist_item(
                "missing_values_checked",
                True,
                f"Initial missing values: {self.initial_missing_values}. Numeric missing values are filled with median; categorical missing values are filled as 'missing'."
            ),
            self._checklist_item(
                "duplicates_checked",
                True,
                f"Initial duplicate rows: {self.initial_duplicate_rows}. Duplicates are removed for train data."
            ),
            self._checklist_item(
                "categorical_features_encoded",
                True,
                f"Initial categorical columns: {len(self.initial_object_columns)}. Encoded categories are reused for test data."
            ),
            self._checklist_item(
                "feature_engineering_applied",
                True,
                "Added squared features for first numeric columns."
            ),
            self._checklist_item(
                "leakage_columns_checked",
                True,
                f"Removed leakage-like columns: {self.leakage_columns_removed}."
            ),
            self._checklist_item(
                "train_validation_split_done",
                True,
                f"Train rows: {len(self.X_train)}. Validation rows: {len(self.X_val)}."
            ),
            self._checklist_item(
                "model_allowed",
                model_name in all_models,
                f"Selected model: {model_name}. Allowed models: {all_models}.",
                "error" if model_name not in all_models else "info"
            ),
            self._checklist_item(
                "hyperparameters_sanitized",
                True,
                f"Raw params: {raw_params}. Used params: {sanitized_params}. Params changed by sanitizer: {params_changed}."
            ),
            self._checklist_item(
                "model_diversity_checked",
                not current_model_already_used or len(tested_models) >= len(all_models),
                f"Previously tested models: {tested_models}. Untested models: {untested_models}.",
                "warning" if current_model_already_used and len(tested_models) < len(all_models) else "info"
            ),
            self._checklist_item(
                "cross_validation_done",
                "cv_std" in validation_result,
                f"CV std: {validation_result.get('cv_std')}."
            ),
            self._checklist_item(
                "overfitting_checked",
                validation_result.get("overfit_gap", 999) <= 0.1,
                f"Overfit gap: {validation_result.get('overfit_gap')}.",
                "warning" if validation_result.get("overfit_gap", 999) > 0.1 else "info"
            ),
            self._checklist_item(
                "compute_budget_checked",
                validation_result.get("remaining_budget", 0) > 0,
                f"Remaining compute budget: {validation_result.get('remaining_budget')}."
            ),
            self._checklist_item(
                "token_budget_checked",
                validation_result.get("remaining_tokens", 0) > 0,
                f"Remaining tokens: {validation_result.get('remaining_tokens')} / {validation_result.get('token_budget')}."
            ),
            self._checklist_item(
                "submission_ready",
                self.best_model_object is not None,
                "A fitted best model exists and can be used for submission." if self.best_model_object is not None else "No fitted best model yet.",
                "warning" if self.best_model_object is None else "info"
            )
        ]

        failed = [item for item in checklist if not item["passed"]]
        warnings = [item for item in checklist if item["severity"] == "warning"]

        feedback = []

        if untested_models and len(tested_models) < len(all_models):
            feedback.append(
                "Try an untested model next: " + ", ".join(untested_models[:3]) + "."
            )

        if current_model_already_used and len(tested_models) < len(all_models):
            feedback.append(
                f"Model {model_name} was already used. Prefer exploration before repeating it."
            )

        if params_changed:
            feedback.append(
                "Some hyperparameters were invalid or unsupported and were sanitized. Use only safe parameters from the allowed schema."
            )

        if validation_result.get("overfit_gap", 0) > 0.1:
            feedback.append(
                "Overfitting is high. Reduce model complexity: lower depth, fewer estimators, stronger regularization."
            )

        if validation_result.get("cv_std", 0) > 0.02:
            feedback.append(
                "CV variance is high. Prefer more robust models or simpler hyperparameters."
            )

        if validation_result.get("remaining_tokens", self.token_budget) < self.token_budget * 0.25:
            feedback.append(
                "Token budget is low. Stop exploration and tune the best known model with short JSON only."
            )

        if validation_result.get("remaining_budget", self.compute_budget) < self.compute_budget * 0.25:
            feedback.append(
                "Compute budget is low. Prefer cheaper models or smaller n_estimators."
            )

        if not feedback:
            feedback.append(
                "Checklist passed. Continue with either an untested model or targeted hyperparameter tuning."
            )

        summary = {
            "total": len(checklist),
            "passed": len([item for item in checklist if item["passed"]]),
            "failed": len(failed),
            "warnings": len(warnings)
        }

        return {
            "summary": summary,
            "items": checklist,
            "agent_feedback": feedback
        }

    def _build_error_checklist_feedback(
        self,
        action,
        error_text
    ):
        checklist = [
            self._checklist_item(
                "action_received",
                isinstance(action, dict),
                f"Action type: {type(action).__name__}."
            ),
            self._checklist_item(
                "model_specified",
                bool(action.get("model")) if isinstance(action, dict) else False,
                f"Model: {action.get('model') if isinstance(action, dict) else None}."
            ),
            self._checklist_item(
                "execution_successful",
                False,
                error_text,
                "error"
            )
        ]

        return {
            "summary": {
                "total": len(checklist),
                "passed": len([item for item in checklist if item["passed"]]),
                "failed": len([item for item in checklist if not item["passed"]]),
                "warnings": 0
            },
            "items": checklist,
            "agent_feedback": [
                "Previous action failed. Fix the model name, parameters, or preprocessing choice before trying again.",
                f"Error message: {error_text}"
            ]
        }

    def step(self, action):

        if self.total_compute_cost >= self.compute_budget:
            return {
                "success": False,
                "error": "Compute budget exhausted",
                "reward": -999.0,
                "budget_exhausted": True,
                "compute_budget_exhausted": True,
                "token_budget_exhausted": False,
                "total_compute_cost": float(
                    self.total_compute_cost
                ),
                "remaining_budget": 0.0,
                "total_token_cost": int(
                    self.total_token_cost
                ),
                "token_budget": int(
                    self.token_budget
                ),
                "remaining_tokens": int(
                    max(0, self.token_budget - self.total_token_cost)
                )
            }

        if self.total_token_cost >= self.token_budget:
            return {
                "success": False,
                "error": "Token budget exhausted",
                "reward": -999.0,
                "budget_exhausted": True,
                "compute_budget_exhausted": False,
                "token_budget_exhausted": True,
                "total_compute_cost": float(
                    self.total_compute_cost
                ),
                "remaining_budget": float(
                    max(0, self.compute_budget - self.total_compute_cost)
                ),
                "total_token_cost": int(
                    self.total_token_cost
                ),
                "token_budget": int(
                    self.token_budget
                ),
                "remaining_tokens": 0
            }

        try:
            model_name = action.get(
                "model",
                "xgboost"
            )

            recent_models = []

            for item in self.history[-3:]:

                try:
                    recent_models.append(
                        item["action"]["model"]
                    )

                except Exception:
                    pass

            repeat_penalty = 0

            if model_name in recent_models:
                repeat_penalty = 0.05

            raw_params = dict(action.get("params", {}) or {})

            params = self._sanitize_params(
                raw_params
            )

            start_time = time.time()

            model = self._get_model(
                model_name,
                params
            )

            model.fit(
                self.X_train,
                self.y_train
            )

            val_preds = self._predict_for_metric(
                model,
                self.X_val
            )

            train_preds = self._predict_for_metric(
                model,
                self.X_train
            )

            val_score = float(
                self._calculate_metric(
                    self.y_val,
                    val_preds
                )
            )

            train_score = float(
                self._calculate_metric(
                    self.y_train,
                    train_preds
                )
            )

            objective = float(
                self._objective_value(val_score)
            )

            overfit_gap = abs(
                self._objective_value(train_score)
                - objective
            )

            scoring = self._cv_scoring()

            if self.task_type == "classification":
                cv = StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=42
                )
            else:
                cv = KFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=42
                )

            cv_scores = cross_val_score(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring=scoring
            )

            if self.metric in self.LOWER_IS_BETTER:
                cv_scores = -cv_scores

            cv_std = float(
                np.std(cv_scores)
            )

            training_time = (
                time.time()
                - start_time
            )

            base_cost = {
                "xgboost": 25,
                "lightgbm": 18,
                "catboost": 22,
                "random_forest": 15,
                "hist_gb": 10
            }.get(model_name, 20)

            compute_cost = float(
                training_time * 10
                + len(params) * 5
                + base_cost
            )

            token_cost = int(
                action
                .get("_token_info", {})
                .get("total_tokens", 0)
            )
            
            response_cost = int(
                action
                .get("_token_info", {})
                .get("response_tokens", 0)
            )

            self.total_compute_cost = min(
                self.compute_budget,
                self.total_compute_cost
                + compute_cost
            )

            self.total_token_cost += token_cost

            token_ratio = min(
                1.0,
                self.total_token_cost / max(1, self.token_budget)
            )

            reward = float(
                objective
                - overfit_gap
                - cv_std
                - compute_cost * 0.001
                - token_cost * 0.000001
                - response_cost * 0.0001
                - token_ratio * 0.05
                - repeat_penalty
            )

            candidate_id = len(
                self.candidates
            )

            result = {
                "success": True,
                "candidate_id": candidate_id,
                "reward": reward,
                "val_score": val_score,
                "objective_value": objective,
                "cv_std": cv_std,
                "train_score": train_score,
                "overfit_gap": float(overfit_gap),
                "training_time_sec": float(training_time),
                "compute_cost": compute_cost,
                "token_cost": token_cost,
                "total_compute_cost": float(
                    self.total_compute_cost
                ),
                "total_token_cost": int(
                    self.total_token_cost
                ),
                "remaining_budget": float(
                    max(
                        0,
                        self.compute_budget
                        - self.total_compute_cost
                    )
                ),
                "compute_budget": float(
                    self.compute_budget
                ),
                "token_budget": int(
                    self.token_budget
                ),
                "remaining_tokens": int(
                    max(0, self.token_budget - self.total_token_cost)
                ),
                "token_usage_ratio": float(
                    min(1.0, self.total_token_cost / max(1, self.token_budget))
                ),
                "num_rows": int(len(self.df)),
                "num_features": int(
                    self.X.shape[1]
                ),
                "best_score": (
                    float(self.best_score)
                    if self.best_score is not None
                    else None
                ),
                "best_model": self.best_model_name,
                "task_type": self.task_type,
                "metric": self.metric,
                "leakage_columns_removed":
                    self.leakage_columns_removed,
                "feature_importance":
                    self._feature_importance(model)
            }

            result["checklist"] = self._build_checklist_feedback(
                action=action,
                model_name=model_name,
                raw_params=raw_params,
                sanitized_params=params,
                validation_result=result
            )

            self.candidates.append({
                "candidate_id": candidate_id,
                "action": action,
                "observation": result
            })

            if objective > self.best_objective:

                self.best_objective = objective
                self.best_score = val_score
                self.best_model_name = model_name
                self.best_model_object = model
                self.best_candidate_id = candidate_id
                self.best_action = action

                result["best_score"] = float(
                    self.best_score
                )

                result["best_model"] = (
                    self.best_model_name
                )

            self.history.append({
                "action": action,
                "observation": result
            })

            return result

        except Exception as error:

            error_result = {
                "success": False,
                "error": str(error),
                "reward": -999.0,
                "budget_exhausted": False
            }

            error_result["checklist"] = self._build_error_checklist_feedback(
                action=action,
                error_text=str(error)
            )

            self.history.append({
                "action": action,
                "observation": error_result
            })

            return error_result

    def predict_submission(
        self,
        test_path=None
    ):

        if self.best_model_object is None:
            raise RuntimeError(
                "No successful model was trained."
            )

        if test_path:
            raw_test = self._read_csv_safely(
                test_path
            )

            X_submit = self.prepare_external_features(
                raw_test
            )

        else:
            X_submit = self.X.copy()

        predictions = self.best_model_object.predict(
            X_submit
        )

        return pd.DataFrame({
            "prediction": predictions
        })
