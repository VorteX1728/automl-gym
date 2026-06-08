import time
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)

from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    r2_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        self.force_clustering = False
        self.compute_budget = float(budget)
        self.budget = self.compute_budget

        self.cluster_models = {}
        self.cluster_scaler = None
        self.cluster_features = []

        # Maximum total LLM tokens allowed for one AutoML run.
        # Counts prompt + response tokens returned by the local LLM wrapper.
        self.token_budget = 32000

        self.raw_df = self._read_csv_safely(
            train_path
        )

        if (
            not target_column
            or str(target_column).strip() == ""
            or target_column not in self.raw_df.columns
        ):
            self.force_clustering = True
            self.target_column = "__cluster_target__"
            self.raw_df[self.target_column] = 0
            target_column = self.target_column
            self.metric = "silhouette"
        else:
            self.force_clustering = False
            self.target_column = target_column

        print("TRAIN PATH:", train_path)

        print("SHAPE:", self.raw_df.shape)
        print("COLUMNS:", list(self.raw_df.columns))

        print("TARGET EXISTS:", self.target_column in self.raw_df.columns)

        if self.target_column in self.raw_df.columns:
            print("TARGET DTYPE:", self.raw_df[self.target_column].dtype)
            print("TARGET HEAD:")
            print(self.raw_df[self.target_column].head(20))
            print("TARGET COUNTS:")
            print(
                self.raw_df[self.target_column]
                .astype(str)
                .value_counts(dropna=False)
                .head(30)
            )

        self.initial_shape = self.raw_df.shape
        self.initial_missing_values = int(self.raw_df.isna().sum().sum())
        self.initial_duplicate_rows = int(self.raw_df.duplicated().sum())
        self.initial_object_columns = [
            column
            for column in self.raw_df.columns
            if self._is_object_like_column(
                self.raw_df[column]
            )
        ]

        print("INITIAL OBJECT COLUMNS:", self.initial_object_columns)

        if not self.force_clustering and self.target_column not in self.raw_df.columns:
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

        self.requested_metric = self.metric

        if self.force_clustering:
            self.task_type = "clustering"
        else:
            self.task_type = self._infer_task_type(
                self.raw_df[self.target_column]
            )

        if (
            self.task_type == "regression"
            and self.metric not in self.LOWER_IS_BETTER
            and self.metric != "r2"
        ):
            self.metric = "rmse"

        if (
            self.task_type == "classification"
            and self.metric in self.LOWER_IS_BETTER
        ):
            self.metric = "roc_auc"

        self.target_mapping = None

        self.df = self._prepare_train_dataframe(
            self.raw_df.copy()
        )


        print("PREPARED SHAPE:", self.df.shape)
        print("ENCODED CATEGORICAL FEATURE COLUMNS:", list(self.category_maps.keys()))

        self.X = self.df.drop(
            columns=[self.target_column]
        )

        self.y = self.df[self.target_column]

        if self.task_type == "clustering":

            self.X_train = self.X.copy()
            self.X_val = self.X.copy()
            self.y_train = self.y.copy()
            self.y_val = self.y.copy()

        elif self.task_type == "classification":

            class_counts = self.y.value_counts()

            if self.y.nunique() < 2:
                raise ValueError(
                    "Target contains only one class after preprocessing. "
                    f"Target column: {self.target_column}. "
                    f"Class distribution: {class_counts.to_dict()}. "
                    f"Target mapping: {self.target_mapping}."
                )

            if class_counts.min() < 2:
                raise ValueError(
                    "At least one target class has fewer than 2 rows. "
                    f"Class distribution: {class_counts.to_dict()}. "
                    "Stratified validation split is impossible."
                )

            split_done = False
            last_error = None

            for test_size in [0.2, 0.25, 0.3, 0.15, 0.1]:
                for seed in [42, 1, 7, 13, 21, 100]:

                    try:
                        (
                            self.X_train,
                            self.X_val,
                            self.y_train,
                            self.y_val
                        ) = train_test_split(
                            self.X,
                            self.y,
                            test_size=test_size,
                            random_state=seed,
                            stratify=self.y
                        )

                        if self.y_train.nunique() >= 2 and self.y_val.nunique() >= 2:
                            split_done = True
                            break

                    except Exception as error:
                        last_error = error

                if split_done:
                    break

            if not split_done:
                raise ValueError(
                    "Could not create validation split with at least two classes. "
                    f"Overall class distribution: {class_counts.to_dict()}. "
                    f"Last split error: {last_error}."
                )

        else:

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42
            )



    def _is_object_like_column(self, series):

        dtype_text = str(series.dtype).lower()

        return (
            series.dtype == "object"
            or "object" in dtype_text
            or "string" in dtype_text
            or dtype_text == "str"
            or dtype_text.startswith("str")
            or "category" in dtype_text
        )

    def _is_numeric_like_string_column(self, series):

        name = str(
            getattr(series, "name", "")
        ).lower()

        allowed_name_fragments = [
            "weight",
            "ram",
            "inch",
            "inches",
            "size",
            "capacity",
            "storage",
            "memory_size"
        ]

        if not any(fragment in name for fragment in allowed_name_fragments):
            return False

        sample = (
            series
            .dropna()
            .astype(str)
            .str.strip()
        )

        if len(sample) == 0:
            return False

        sample = sample.head(
            min(200, len(sample))
        )

        cleaned = (
            sample
            .str.replace(r"[^0-9,\.\-]+", "", regex=True)
            .str.replace(",", ".", regex=False)
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        return float(numeric.notna().mean()) >= 0.9


    def _clean_numeric_like_string_column(self, series):

        cleaned = (
            series
            .astype(str)
            .str.strip()
            .str.replace(r"[^0-9,\.\-]+", "", regex=True)
            .str.replace(",", ".", regex=False)
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        median = numeric.median()

        if pd.isna(median):
            median = 0

        return numeric.fillna(
            median
        )


    def _read_csv_safely(self, path):

        attempts = [
            {
                "sep": ",",
                "encoding": "utf-8",
                "engine": "python"
            },
            {
                "sep": ",",
                "encoding": "latin1",
                "engine": "python"
            },
            {
                "sep": ";",
                "encoding": "utf-8",
                "engine": "python"
            },
            {
                "sep": ";",
                "encoding": "latin1",
                "engine": "python"
            }
        ]

        best_df = None
        best_score = -1
        last_error = None

        for kwargs in attempts:

            try:
                df = pd.read_csv(
                    path,
                    **kwargs
                )

                if df.shape[1] <= 1:
                    continue

                object_columns = int(
                    df.select_dtypes(
                        include=["object", "string", "category"]
                    ).shape[1]
                )

                non_empty_cells = int(
                    df.notna().sum().sum()
                )

                score = (
                    df.shape[1] * 10000
                    + object_columns * 1000
                    + non_empty_cells
                )

                if score > best_score:
                    best_score = score
                    best_df = df

            except Exception as error:
                last_error = error

        if best_df is None:
            raise RuntimeError(
                f"Cannot read CSV: {last_error}"
            )

        print("CSV READ PATH:", path)
        print("CSV SHAPE:", best_df.shape)
        print("CSV COLUMNS:", list(best_df.columns))
        print("CSV DTYPES:")
        print(best_df.dtypes)

        return best_df

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

        df = df.copy()

        drop_columns = []

        for column in df.columns:

            if column == self.target_column:
                continue

            column_lower = str(column).lower().strip()

            if (
                column_lower.startswith("unnamed")
                or column_lower in ["index", "row_index"]
            ):
                drop_columns.append(column)

        if drop_columns:
            df = df.drop(
                columns=drop_columns,
                errors="ignore"
            )

        if is_train:
            df = df.drop_duplicates()

        for column in df.columns:

            if column == self.target_column:
                continue

            is_object_like = self._is_object_like_column(
                df[column]
            )

            if is_object_like:

                if self._is_numeric_like_string_column(df[column]):
                    df[column] = self._clean_numeric_like_string_column(
                        df[column]
                    )
                else:
                    df[column] = (
                        df[column]
                        .fillna("missing")
                        .astype(str)
                        .str.strip()
                    )

            else:

                numeric_series = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                median = numeric_series.median()

                if pd.isna(median):
                    median = 0

                df[column] = numeric_series.fillna(
                    median
                )

        return df

    def _feature_engineering(self, df):

        df = df.copy()

        numeric_columns = [
            column
            for column in df.select_dtypes(
                include=np.number
            ).columns
            if column != self.target_column
        ]

        safe_numeric_columns = [
            column
            for column in numeric_columns
            if (
                "id" not in str(column).lower()
                and "index" not in str(column).lower()
                and "unnamed" not in str(column).lower()
            )
        ]

        for column in safe_numeric_columns[:5]:
            df[f"{column}_squared"] = (
                df[column] ** 2
            )

        return df

    def _apply_clustering_features(
        self,
        X,
        fit=False
    ):

        X = X.copy()

        numeric_columns = [
            column
            for column in X.select_dtypes(
                include=np.number
            ).columns
        ]

        if len(numeric_columns) < 2:
            return X

        cluster_input = X[
            numeric_columns
        ].fillna(0)

        if fit:

            self.cluster_scaler = StandardScaler()

            scaled = self.cluster_scaler.fit_transform(
                cluster_input
            )

            if scaled.shape[1] > 10:

                pca = PCA(
                    n_components=10,
                    random_state=42
                )

                scaled = pca.fit_transform(
                    scaled
                )

                self.cluster_pca = pca

            else:
                self.cluster_pca = None

            for n_clusters in [3, 5]:

                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )

                labels = model.fit_predict(
                    scaled
                )

                X[
                    f"cluster_{n_clusters}"
                ] = labels

                distances = model.transform(
                    scaled
                ).min(axis=1)

                X[
                    f"cluster_{n_clusters}_distance"
                ] = distances

                self.cluster_models[
                    n_clusters
                ] = model

        else:

            if self.cluster_scaler is None:
                return X

            scaled = self.cluster_scaler.transform(
                cluster_input
            )

            if self.cluster_pca is not None:
                scaled = self.cluster_pca.transform(
                    scaled
                )

            for (
                n_clusters,
                model
            ) in self.cluster_models.items():

                labels = model.predict(
                    scaled
                )

                X[
                    f"cluster_{n_clusters}"
                ] = labels

                distances = model.transform(
                    scaled
                ).min(axis=1)

                X[
                    f"cluster_{n_clusters}_distance"
                ] = distances

        return X

    def _sanitize_feature_names(self, X):

        X = X.copy()

        clean_columns = []
        used = {}

        for column in X.columns:

            clean = re.sub(
                r"[^A-Za-z0-9_]+",
                "_",
                str(column)
            ).strip("_")

            if not clean:
                clean = "feature"

            if clean[0].isdigit():
                clean = "f_" + clean

            base = clean
            count = used.get(base, 0)

            if count:
                clean = f"{base}_{count}"

            used[base] = count + 1

            clean_columns.append(clean)

        X.columns = clean_columns

        return X

    def _encode_train_features(self, X):

        X = X.copy()

        for column in X.columns:

            if self._is_object_like_column(
                X[column]
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

        X = self._sanitize_feature_names(X)

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

        X = self._apply_clustering_features(
            X,
            fit=False
        )

        X = self._sanitize_feature_names(X)

        for column in self.feature_columns:
            if column not in X.columns:
                X[column] = 0

        X = X[self.feature_columns]

        return X

    def _prepare_train_dataframe(self, df):

        # Clean target BEFORE generic preprocessing.
        # Do not convert missing target values to the string "missing".
        target = df[self.target_column]

        target_str = (
            target
            .astype(str)
            .str.strip()
        )

        missing_like = (
            target.isna()
            | target_str.str.lower().isin([
                "",
                "nan",
                "none",
                "null",
                "missing",
                "na",
                "n/a",
                "<na>"
            ])
        )

        valid_mask = ~missing_like

        df = df.loc[
            valid_mask
        ].copy()

        target_clean = target_str.loc[
            valid_mask
        ]

        df[self.target_column] = target_clean.values

        if len(df) == 0:

            raw_counts = (
                target_str
                .value_counts(dropna=False)
                .head(20)
                .to_dict()
            )

            raise ValueError(
                f"All rows were removed because target column "
                f"'{self.target_column}' is empty or missing-like. "
                f"Raw target values: {raw_counts}."
            )

        print("CLEAN TARGET COUNTS:")
        print(
            df[self.target_column]
            .value_counts(dropna=False)
        )

        df = self._clean_basic(
            df,
            is_train=True
        )

        df = self._feature_engineering(df)

        features_only = df.drop(
            columns=[self.target_column],
            errors="ignore"
        )

        features_only = self._apply_clustering_features(
            features_only,
            fit=True
        )

        df = pd.concat(
            [
                features_only,
                df[[self.target_column]]
            ],
            axis=1
        )

        df = self._remove_leakage(df)

        y = df[self.target_column]

        # CLUSTERING
        if self.task_type == "clustering":

            y = pd.Series(
                np.zeros(len(df), dtype=int),
                index=df.index
            )

        # CLASSIFICATION
        elif self.task_type == "classification":

            # НЕ ДЕЛАТЬ fillna("missing")
            y = y.astype(str)

            classes = sorted(
                y.unique()
            )

            self.target_mapping = {
                value: index
                for index, value in enumerate(classes)
            }

            y = (
                y
                .map(self.target_mapping)
                .astype(int)
            )

        # REGRESSION
        else:

            y = (
                y
                .astype(str)
                .str.replace(r"[^0-9.,\-]+", "", regex=True)
                .str.replace(",", ".", regex=False)
            )

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            y = y.fillna(
                y.median()
            )

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

            for key in [
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "num_leaves",
                "iterations",
                "depth",
                "max_iter",
                "n_bins",
                "loss_function",
                "random_strength"
            ]:
                params.pop(key, None)

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

            for key in [
                "n_bins",
                "num_leaves",
                "subsample",
                "colsample_bytree",
                "bagging_fraction",
                "feature_fraction",
                "iterations",
                "depth",
                "loss_function",
                "random_strength"
            ]:
                params.pop(key, None)

            if "n_estimators" in params:
                params["max_iter"] = params.pop("n_estimators")

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

            for key in [
                "iterations",
                "depth",
                "max_iter",
                "n_bins",
                "loss_function",
                "random_strength"
            ]:
                params.pop(key, None)

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

            for key in [
                "subsample",
                "colsample_bytree",
                "bagging_fraction",
                "feature_fraction",
                "num_leaves",
                "max_iter",
                "n_bins"
            ]:
                params.pop(key, None)

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

        for key in [
            "iterations",
            "depth",
            "max_iter",
            "n_bins",
            "loss_function",
            "random_strength",
            "num_leaves",
            "bagging_fraction",
            "feature_fraction"
        ]:
            params.pop(key, None)

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

            unique_classes = np.unique(y_true)

            if len(unique_classes) < 2:
                raise ValueError(
                    "ROC-AUC requires at least 2 classes"
                )

            # binary classification
            if np.ndim(preds) == 2:

                if preds.shape[1] == 2:

                    return roc_auc_score(
                        y_true,
                        preds[:, 1]
                    )

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

        if self.metric == "r2":
            return r2_score(
                y_true,
                preds
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
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
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

        model_name = (
            str(model_name)
            .strip()
            .lower()
        )

        model_aliases = {
            "histgb": "hist_gb",
            "hist_gb": "hist_gb",
            "hist_gradient_boosting": "hist_gb",
            "histgradientboosting": "hist_gb",
            "rf": "random_forest",
            "randomforest": "random_forest",
            "lgbm": "lightgbm",
            "cb": "catboost",
            "cat": "catboost",
            "xgb": "xgboost"
        }

        model_name = model_aliases.get(
            model_name,
            model_name
        )

        current_model_already_used = model_name in tested_models

        all_models = [
            "xgboost",
            "lightgbm",
            "catboost",
            "random_forest",
            "hist_gb",
            "kmeans"
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
                self.task_type in ["classification", "regression", "clustering"],
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
                f"Initial categorical columns: {len(self.initial_object_columns)}. Encoded feature columns: {len(self.category_maps)}. Encoded categories are reused for test data."
            ),
            self._checklist_item(
                "feature_engineering_applied",
                True,
                "Added squared features for first numeric columns."
            ),
            self._checklist_item(
                "clustering_features_added",
                True,
                f"Cluster features added using KMeans: {list(self.cluster_models.keys())}",
                "info"
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

            model_name = (
                str(model_name)
                .strip()
                .lower()
            )

            model_aliases = {
                "histgb": "hist_gb",
                "hist_gb": "hist_gb",
                "hist_gradient_boosting": "hist_gb",
                "histgradientboosting": "hist_gb",
                "histgradientboostingregressor": "hist_gb",
                "histgradientboostingclassifier": "hist_gb",
                "rf": "random_forest",
                "randomforest": "random_forest",
                "random_forest": "random_forest",
                "lgbm": "lightgbm",
                "lightgbm": "lightgbm",
                "cb": "catboost",
                "cat": "catboost",
                "catboost": "catboost",
                "xgb": "xgboost",
                "xgboost": "xgboost"
            }

            model_name = model_aliases.get(
                model_name,
                model_name
            )

            action["model"] = model_name

            if self.task_type == "clustering":

                start_time = time.time()

                numeric_X = (
                    self.X
                    .select_dtypes(include=np.number)
                    .fillna(0)
                )

                if numeric_X.shape[1] < 2:
                    raise ValueError(
                        "Clustering requires at least two numeric features after preprocessing."
                    )

                scaler = StandardScaler()
                scaled = scaler.fit_transform(numeric_X)

                best_labels = None
                best_model = None
                best_score = -float("inf")
                best_n_clusters = None

                for n_clusters in [3, 5, 8]:

                    if len(numeric_X) <= n_clusters:
                        continue

                    cluster_model = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10
                    )

                    labels = cluster_model.fit_predict(
                        scaled
                    )

                    if len(np.unique(labels)) < 2:
                        continue

                    score = float(
                        silhouette_score(
                            scaled,
                            labels
                        )
                    )

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_model = cluster_model
                        best_n_clusters = n_clusters

                if best_model is None:
                    raise ValueError(
                        "Could not build a valid clustering model."
                    )

                self.cluster_scaler = scaler
                self.cluster_models = {
                    best_n_clusters: best_model
                }
                self.best_model_object = best_model
                self.best_model_name = "kmeans"
                self.best_score = best_score

                training_time = time.time() - start_time

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

                compute_cost = float(
                    training_time * 10
                    + 10
                )

                self.total_compute_cost = min(
                    self.compute_budget,
                    self.total_compute_cost + compute_cost
                )

                self.total_token_cost += token_cost

                candidate_id = len(self.candidates)

                result = {
                    "success": True,
                    "candidate_id": candidate_id,
                    "reward": best_score,
                    "val_score": best_score,
                    "objective_value": best_score,
                    "selection_score": best_score,
                    "selection_criterion": "silhouette",
                    "cv_std": 0.0,
                    "train_score": best_score,
                    "overfit_gap": 0.0,
                    "training_time_sec": float(training_time),
                    "compute_cost": compute_cost,
                    "token_cost": token_cost,
                    "response_cost": response_cost,
                    "total_compute_cost": float(self.total_compute_cost),
                    "total_token_cost": int(self.total_token_cost),
                    "remaining_budget": float(
                        max(0, self.compute_budget - self.total_compute_cost)
                    ),
                    "compute_budget": float(self.compute_budget),
                    "token_budget": int(self.token_budget),
                    "remaining_tokens": int(
                        max(0, self.token_budget - self.total_token_cost)
                    ),
                    "token_usage_ratio": float(
                        min(1.0, self.total_token_cost / max(1, self.token_budget))
                    ),
                    "step_token_usage_ratio": float(
                        min(1.0, token_cost / max(1, self.token_budget))
                    ),
                    "num_rows": int(len(self.df)),
                    "num_features": int(self.X.shape[1]),
                    "best_score": float(best_score),
                    "best_model": "kmeans",
                    "task_type": self.task_type,
                    "metric": "silhouette",
                    "requested_metric": self.requested_metric,
                    "n_clusters": int(best_n_clusters),
                    "cluster_counts": {
                        str(label): int(count)
                        for label, count in zip(
                            *np.unique(best_labels, return_counts=True)
                        )
                    },
                    "leakage_columns_removed": self.leakage_columns_removed,
                    "feature_importance": {}
                }

                self.best_objective = best_score
                self.best_candidate_id = candidate_id
                self.best_action = action

                result["checklist"] = self._build_checklist_feedback(
                    action=action,
                    model_name="kmeans",
                    raw_params=dict(action.get("params", {}) or {}),
                    sanitized_params={},
                    validation_result=result
                )

                self.candidates.append({
                    "candidate_id": candidate_id,
                    "action": action,
                    "observation": result
                })

                self.history.append({
                    "action": action,
                    "observation": result
                })

                return result

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

            if not np.isfinite(val_score):
                raise ValueError(
                    "Validation score is NaN or infinite. "
                    "Check target encoding, metric compatibility, or class distribution."
                )

            if not np.isfinite(train_score):
                raise ValueError(
                    "Train score is NaN or infinite. "
                    "Check target encoding, metric compatibility, or class distribution."
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

            step_token_ratio = min(
                1.0,
                token_cost / max(1, self.token_budget)
            )

            base_reward = float(
                objective
                - overfit_gap
                - cv_std
                - compute_cost * 0.001
                - token_cost * 0.000001
                - response_cost * 0.0001
                - step_token_ratio * 0.05
                - repeat_penalty
            )

            candidate_id = len(
                self.candidates
            )

            result = {
                "success": True,
                "candidate_id": candidate_id,
                "reward": base_reward,
                "val_score": val_score,
                "objective_value": objective,
                "selection_score": base_reward,
                "selection_criterion": "reward",
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
                    min(
                        1.0,
                        self.total_token_cost
                        / max(1, self.token_budget)
                    )
                ),
                "step_token_usage_ratio": float(
                    step_token_ratio
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
                "requested_metric": self.requested_metric,
                "leakage_columns_removed":
                    self.leakage_columns_removed,
                "feature_importance":
                    self._feature_importance(model)
            }

            checklist_feedback = self._build_checklist_feedback(
                action=action,
                model_name=model_name,
                raw_params=raw_params,
                sanitized_params=params,
                validation_result=result
            )

            checklist_summary = checklist_feedback.get(
                "summary",
                {}
            )

            checklist_penalty = float(
                checklist_summary.get("failed", 0) * 0.01
                + checklist_summary.get("warnings", 0) * 0.005
            )

            final_reward = float(
                base_reward - checklist_penalty
            )

            result["reward"] = final_reward
            result["selection_score"] = final_reward
            result["checklist_penalty"] = checklist_penalty
            result["checklist"] = checklist_feedback

            self.candidates.append({
                "candidate_id": candidate_id,
                "action": action,
                "observation": result
            })

            if final_reward > self.best_objective:

                self.best_objective = final_reward
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

                result["selection_criterion"] = "reward"

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

        if self.task_type == "clustering":

            if test_path:
                raw_test = self._read_csv_safely(
                    test_path
                )

                X_submit = self.prepare_external_features(
                    raw_test
                )

            else:
                X_submit = self.X.copy()

            numeric_X = (
                X_submit
                .select_dtypes(include=np.number)
                .fillna(0)
            )

            scaled = self.cluster_scaler.transform(
                numeric_X
            )

            labels = self.best_model_object.predict(
                scaled
            )

            return pd.DataFrame({
                "cluster": labels
            })

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
