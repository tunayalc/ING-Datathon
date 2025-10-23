import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Kütüphaneler (GPU için) ---
HAVE_CATBOOST = True
try:
    from catboost import CatBoostClassifier
except Exception:
    HAVE_CATBOOST = False

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

RANDOM_STATE = 42
WINDOW_SIZES = [3, 6, 12]
TOP_DECILE = 0.10
MONTH_LAGS = [1, 2, 3, 4, 5, 6]
RECENT_WINDOW = 3
RECENT_LAG = 3
RECENT_ACTIVITY_BASE_COLS = [
    "total_txn_amt",
    "total_txn_cnt",
    "mobile_eft_all_amt",
    "mobile_eft_all_cnt",
    "cc_transaction_all_amt",
    "cc_transaction_all_cnt",
    "amt_per_active_product",
    "cnt_per_active_product",
]

# =============================
#  Metric helpers
# =============================

def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_pred)
    return 2.0 * auc - 1.0


def lift_at_k(y_true: np.ndarray, y_pred: np.ndarray, top_percent: float = TOP_DECILE) -> float:
    order = np.argsort(-y_pred)
    cutoff = max(1, int(np.ceil(len(y_true) * top_percent)))
    top_true = y_true[order][:cutoff]
    baseline_rate = y_true.mean()
    top_rate = top_true.mean()
    if baseline_rate == 0:
        return np.nan
    return float(top_rate / baseline_rate)


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, top_percent: float = TOP_DECILE) -> float:
    order = np.argsort(-y_pred)
    cutoff = max(1, int(np.ceil(len(y_true) * top_percent)))
    top_true = y_true[order][:cutoff]
    positives = y_true.sum()
    if positives == 0:
        return np.nan
    return float(top_true.sum() / positives)


def custom_weighted_score(gini: float, lift10: float, recall10: float) -> float:
    return 0.40 * gini + 0.30 * lift10 + 0.30 * recall10


# =============================
#  Feature engineering
# =============================

def compute_history_features(history: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.to_period("M").dt.to_timestamp()
    hist = hist.sort_values(["cust_id", "date"]).reset_index(drop=True)

    numeric_cols = [
        "mobile_eft_all_cnt",
        "mobile_eft_all_amt",
        "cc_transaction_all_amt",
        "cc_transaction_all_cnt",
        "active_product_category_nbr",
    ]

    for col in numeric_cols:
        hist[col] = hist[col].fillna(0.0)

    # türetilmişler
    hist["total_txn_cnt"] = hist["mobile_eft_all_cnt"] + hist["cc_transaction_all_cnt"]
    hist["total_txn_amt"] = hist["mobile_eft_all_amt"] + hist["cc_transaction_all_amt"]
    hist["mobile_avg_ticket"] = np.where(hist["mobile_eft_all_cnt"] > 0, hist["mobile_eft_all_amt"] / hist["mobile_eft_all_cnt"], 0.0)
    hist["cc_avg_ticket"] = np.where(hist["cc_transaction_all_cnt"] > 0, hist["cc_transaction_all_amt"] / hist["cc_transaction_all_cnt"], 0.0)
    hist["overall_avg_ticket"] = np.where(hist["total_txn_cnt"] > 0, hist["total_txn_amt"] / hist["total_txn_cnt"], 0.0)
    hist["amt_per_active_product"] = np.where(hist["active_product_category_nbr"] > 0, hist["total_txn_amt"] / hist["active_product_category_nbr"], 0.0)
    hist["cnt_per_active_product"] = np.where(hist["active_product_category_nbr"] > 0, hist["total_txn_cnt"] / hist["active_product_category_nbr"], 0.0)

    derived_numeric_cols = [
        "total_txn_cnt",
        "total_txn_amt",
        "mobile_avg_ticket",
        "cc_avg_ticket",
        "overall_avg_ticket",
        "amt_per_active_product",
        "cnt_per_active_product",
    ]
    numeric_cols.extend(derived_numeric_cols)

    # zaman farkları
    hist["months_since_prev"] = hist.groupby("cust_id")["date"].diff().dt.days.div(30.0).fillna(0.0)
    hist["months_since_first"] = (hist["date"] - hist.groupby("cust_id")["date"].transform("min")).dt.days.div(30.0)
    hist["history_obs_cnt"] = hist.groupby("cust_id").cumcount() + 1

    # rolling / lag / delta + **EWMA/EWMSTD**
    for col in numeric_cols:
        g = hist.groupby("cust_id")[col]
        for lag in MONTH_LAGS:
            lag_values = g.shift(lag).fillna(0.0)
            hist[f"{col}_lag{lag}"] = lag_values
            hist[f"{col}_delta{lag}"] = hist[col] - lag_values
            hist[f"{col}_ratio{lag}"] = hist[col] / (lag_values.abs() + 1e-3)
        cumsum = g.cumsum(); cumcnt = hist.groupby("cust_id").cumcount() + 1
        hist[f"{col}_cumsum"] = cumsum
        hist[f"{col}_cummean"] = cumsum / cumcnt
        for w in windows:
            r = g.rolling(window=w, min_periods=1)
            hist[f"{col}_sum_win{w}"] = r.sum().reset_index(level=0, drop=True)
            hist[f"{col}_mean_win{w}"] = r.mean().reset_index(level=0, drop=True)
            hist[f"{col}_std_win{w}"] = r.std().reset_index(level=0, drop=True)
        # EWMA / EWMSTD (yakın geçmişi daha fazla ağırlıkla)
        hist[f"{col}_ewm_mean_a05"] = g.apply(lambda s: s.ewm(alpha=0.5, adjust=False).mean()).reset_index(level=0, drop=True)
        hist[f"{col}_ewm_std_a05"] = g.apply(lambda s: s.ewm(alpha=0.5, adjust=False).std()).reset_index(level=0, drop=True).fillna(0.0)

    # kanal payları + HHI (korundu)
    for amt_col, cnt_col in [("mobile_eft_all_amt", "mobile_eft_all_cnt"), ("cc_transaction_all_amt", "cc_transaction_all_cnt")]:
        hist[amt_col] = hist[amt_col].fillna(0.0)
        hist[cnt_col] = hist[cnt_col].fillna(0.0)
    total_amt = hist["total_txn_amt"].fillna(0.0)
    total_cnt = hist["total_txn_cnt"].fillna(0.0)
    hist["mobile_amt_share"] = np.clip(hist["mobile_eft_all_amt"] / (total_amt.replace(0, np.nan)), 0.0, 1.0).fillna(0.0)
    hist["cc_amt_share"] = np.clip(hist["cc_transaction_all_amt"] / (total_amt.replace(0, np.nan)), 0.0, 1.0).fillna(0.0)
    hist["mobile_cnt_share"] = np.clip(hist["mobile_eft_all_cnt"] / (total_cnt.replace(0, np.nan)), 0.0, 1.0).fillna(0.0)
    hist["cc_cnt_share"] = np.clip(hist["cc_transaction_all_cnt"] / (total_cnt.replace(0, np.nan)), 0.0, 1.0).fillna(0.0)
    hist["channel_hhi_amt"] = hist["mobile_amt_share"] ** 2 + hist["cc_amt_share"] ** 2
    hist["channel_hhi_cnt"] = hist["mobile_cnt_share"] ** 2 + hist["cc_cnt_share"] ** 2

    # tipleri sıkılaştır
    for col in hist.columns:
        if col not in {"cust_id", "date"} and pd.api.types.is_numeric_dtype(hist[col]):
            hist[col] = hist[col].astype(np.float32)
    return hist


def build_dataset(reference_df: pd.DataFrame, history_features: pd.DataFrame, customers: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = reference_df.copy()
    df["ref_date"] = pd.to_datetime(df["ref_date"]).dt.to_period("M").dt.to_timestamp()

    merged = df.merge(history_features, how="left", left_on=["cust_id", "ref_date"], right_on=["cust_id", "date"]).drop(columns=["date"], errors="ignore")
    merged = merged.merge(customers, how="left", on="cust_id", suffixes=("", "_customer"))

    merged["ref_year"] = merged["ref_date"].dt.year
    merged["ref_month"] = merged["ref_date"].dt.month
    merged["ref_quarter"] = merged["ref_date"].dt.quarter
    merged["ref_month_idx"] = merged["ref_year"] * 12 + merged["ref_month"]

    categorical_cols = ["gender", "province", "religion", "work_type", "work_sector"]
    for col in categorical_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("Unknown").astype(str)

    for col in ["age", "tenure"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())
    if "tenure" in merged.columns:
        merged["tenure_years"] = merged["tenure"] / 12.0
        merged["tenure_log"] = np.log1p(merged["tenure"].clip(lower=0))
    else:
        merged["tenure_years"] = 0.0
        merged["tenure_log"] = 0.0
    if "age" in merged.columns:
        merged["age_log"] = np.log1p(merged["age"].clip(lower=0))
        merged["age_bucket"] = pd.cut(merged["age"], [0,25,35,45,55,65,200], labels=["<=25","26-35","36-45","46-55","56-65","65+"], right=False, include_lowest=True).astype(str)
        if "age_bucket" not in categorical_cols:
            categorical_cols.append("age_bucket")
    else:
        merged["age_log"] = 0.0
        merged["age_bucket"] = "Unknown"
        if "age_bucket" not in categorical_cols:
            categorical_cols.append("age_bucket")

    # numeric doldurma
    for col in merged.select_dtypes(include=["number"]).columns:
        if col not in {"cust_id", "churn"}:
            merged[col] = merged[col].fillna(0.0)

    feature_cols = [c for c in merged.columns if c not in {"cust_id", "ref_date", "churn"}]
    return merged, categorical_cols, feature_cols


# =============================
#  OOF Target Encoding
# =============================
from sklearn.model_selection import StratifiedKFold

def oof_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, target: np.ndarray, cat_cols: List[str], folds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    skf = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    train_df = train_df.copy(); test_df = test_df.copy()
    global_mean = float(target.mean())
    te_cols: List[str] = []
    for col in cat_cols:
        oof = np.zeros(len(train_df), dtype=float)
        for tr, va in skf.split(train_df, target):
            m = train_df.iloc[tr].groupby(col)["churn"].mean()
            oof[va] = train_df.iloc[va][col].map(m).fillna(global_mean).values
        train_df[f"te_{col}"] = oof
        m_full = train_df.groupby(col)["churn"].mean()
        test_df[f"te_{col}"] = test_df[col].map(m_full).fillna(global_mean).values
        te_cols.append(f"te_{col}")
    return train_df, test_df, te_cols


# =============================
#  Modeling (GPU aware)
# =============================

def stratified_parameter_search(model_name: str, param_grid: List[Dict], X: pd.DataFrame, y: np.ndarray, scale_pos_weight: float, cat_features: Optional[List[str]] = None, folds: int = 3, groups: Optional[np.ndarray] = None) -> Dict:
    best_score = -np.inf
    best_params: Optional[Dict] = None
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)

    for params in param_grid:
        fold_scores: List[float] = []
        for tr, va in split_iter:
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            if model_name == "catboost" and HAVE_CATBOOST:
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    od_type="Iter",
                    od_wait=100,
                    scale_pos_weight=scale_pos_weight,
                    task_type="GPU",
                    **params,
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features, use_best_model=True)
                preds = model.predict_proba(X_va)[:, 1]
            elif model_name == "lightgbm":
                model = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=5000,
                    random_state=RANDOM_STATE,
                    scale_pos_weight=scale_pos_weight,
                    device_type="gpu",  # LightGBM >= 4
                    **params,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(200)],
                    categorical_feature=cat_features if cat_features else "auto",
                )
                preds = model.predict_proba(X_va, num_iteration=model.best_iteration_ or model.n_estimators)[:, 1]
            else:
                continue
            fold_scores.append(roc_auc_score(y_va, preds))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    if best_params is None:
        raise RuntimeError(f"No best params identified for {model_name}")
    return {"params": best_params, "score": best_score}


def cross_validate_catboost(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, params: Dict, cat_features: List[str], scale_pos_weight: float, folds: int = 5, groups: Optional[np.ndarray] = None) -> Dict:
    if not HAVE_CATBOOST:
        return {"oof": np.zeros(len(y)), "test": np.zeros(len(X_test)), "importance": pd.Series(0, index=X.columns), "metrics": {"mean_auc": 0.0}}
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(X, y))
    oof = np.zeros(len(y)); test_preds = [] ; imps = []; fold_aucs = []
    for fold, (tr, va) in enumerate(split_iter, 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]; y_tr, y_va = y[tr], y[va]
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_seed=RANDOM_STATE+fold, verbose=False, od_type="Iter", od_wait=150, scale_pos_weight=scale_pos_weight, task_type="GPU", **params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features, use_best_model=True)
        oof[va] = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, oof[va]); fold_aucs.append(fold_auc)
        test_preds.append(model.predict_proba(X_test)[:, 1])
        imps.append(model.get_feature_importance(type="FeatureImportance"))
        print(f"[CatBoost] Fold {fold} AUC: {fold_auc:.5f}")
    importance = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
    return {"oof": oof, "test": np.mean(test_preds, axis=0), "importance": importance, "metrics": {"mean_auc": float(np.mean(fold_aucs))}}


def cross_validate_lightgbm(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, params: Dict, scale_pos_weight: float, categorical_cols: List[str], folds: int = 5, groups: Optional[np.ndarray] = None) -> Dict:
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(X, y))
    oof = np.zeros(len(y)); test_preds = []; imps = []; fold_aucs = []
    for fold, (tr, va) in enumerate(split_iter, 1):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy(); y_tr, y_va = y[tr], y[va]
        model = lgb.LGBMClassifier(objective="binary", n_estimators=5000, random_state=RANDOM_STATE+fold, scale_pos_weight=scale_pos_weight, device_type="gpu", **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=[lgb.early_stopping(200)], categorical_feature=categorical_cols if categorical_cols else "auto")
        best_iter = model.best_iteration_ or model.n_estimators
        oof[va] = model.predict_proba(X_va, num_iteration=best_iter)[:, 1]
        fold_auc = roc_auc_score(y_va, oof[va]); fold_aucs.append(fold_auc)
        test_preds.append(model.predict_proba(X_test, num_iteration=best_iter)[:, 1])
        imps.append(model.feature_importances_)
        print(f"[LightGBM] Fold {fold} AUC: {fold_auc:.5f}")
    importance = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
    return {"oof": oof, "test": np.mean(test_preds, axis=0), "importance": importance, "metrics": {"mean_auc": float(np.mean(fold_aucs))}}


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_pred)
    gini = normalized_gini(y_true, y_pred)
    lift10 = lift_at_k(y_true, y_pred, TOP_DECILE)
    recall10 = recall_at_k(y_true, y_pred, TOP_DECILE)
    score = custom_weighted_score(gini, lift10, recall10)
    print(f"[{name}] AUC: {auc:.5f} | Gini: {gini:.5f} | Lift@10%: {lift10:.5f} | Recall@10%: {recall10:.5f} | Custom: {score:.5f}")
    return {"auc": auc, "gini": gini, "lift10": lift10, "recall10": recall10, "custom_score": score}


# =============================
#  Main
# =============================

def main() -> None:
    # --- Veri yolları ---
    default_dir = Path(r"C:\\Users\\ytuna\\OneDrive\\Masaüstü\\ing-hubs-turkiye-datathon")
    data_dir = default_dir if default_dir.exists() else Path(".")

    train_path = data_dir / "referance_data.csv"   # paket dosya adı böyle
    test_path = data_dir / "referance_data_test.csv"
    customers_path = data_dir / "customers.csv"
    history_path = data_dir / "customer_history.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, parse_dates=["ref_date"])  # label: churn
    test_df = pd.read_csv(test_path, parse_dates=["ref_date"])    
    customers_df = pd.read_csv(customers_path)
    history_df = pd.read_csv(history_path)

    target = train_df["churn"].astype(int).values
    print(f"Training samples: {len(train_df)} | Positive ratio: {target.mean():.4f}")

    print("Engineering historical features...")
    history_features = compute_history_features(history_df, WINDOW_SIZES)

    print("Building datasets...")
    train_merged, cat_cols, feature_cols = build_dataset(train_df, history_features, customers_df)
    test_merged, _, _ = build_dataset(test_df, history_features, customers_df)

    # ==== OOF Target Encoding (leakage'siz) ====
    print("OOF target encoding...")
    te_cat_cols = [c for c in ["gender","province","religion","work_type","work_sector","age_bucket"] if c in train_merged.columns]
    train_merged["churn"] = target
    train_merged, test_merged, te_cols = oof_target_encode(train_merged, test_merged, target, te_cat_cols, folds=5)
    feature_cols = feature_cols + te_cols

    # ==== Tasarım gereği aynı kolonlar ====
    X = train_merged[feature_cols].copy()
    X_test = test_merged[feature_cols].copy()

    # kategorikler: LGBM için category dtype (CatBoost doğrudan handle ediyor)
    for col in te_cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            all_vals = pd.concat([X[col], X_test[col]], axis=0)
            X[col] = pd.Categorical(X[col], categories=all_vals.unique())
            X_test[col] = pd.Categorical(X_test[col], categories=all_vals.unique())

    scale_pos_weight = float((len(target) - target.sum()) / max(1, target.sum()))
    print(f"Using scale_pos_weight: {scale_pos_weight:.4f}")

    groups = train_merged["cust_id"].values

    # ==== Parametre gridleri (hafif) ====
    catboost_param_grid = [
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "iterations": 2000},
        {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5.0, "iterations": 3000},
    ]
    lightgbm_param_grid = [
        {"num_leaves": 48, "learning_rate": 0.04, "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 1, "min_child_samples": 30, "reg_lambda": 2.0},
        {"num_leaves": 64, "learning_rate": 0.03, "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1, "min_child_samples": 25, "reg_lambda": 1.5},
    ]

    print("Searching CatBoost parameters (GPU)...")
    catboost_search = stratified_parameter_search("catboost", catboost_param_grid, X, target, scale_pos_weight=scale_pos_weight, cat_features=[c for c in te_cat_cols if c in X.columns], folds=3, groups=groups) if HAVE_CATBOOST else {"params": {}}

    print("Searching LightGBM parameters (GPU)...")
    lightgbm_search = stratified_parameter_search("lightgbm", lightgbm_param_grid, X, target, scale_pos_weight=scale_pos_weight, cat_features=[c for c in te_cat_cols if c in X.columns], folds=3, groups=groups)

    print("Running CatBoost cross-validation...")
    catboost_cv = cross_validate_catboost(X, target, X_test, params=catboost_search.get("params", {}), cat_features=[c for c in te_cat_cols if c in X.columns], scale_pos_weight=scale_pos_weight, folds=5, groups=groups)

    print("Running LightGBM cross-validation...")
    lightgbm_cv = cross_validate_lightgbm(X, target, X_test, params=lightgbm_search["params"], scale_pos_weight=scale_pos_weight, categorical_cols=te_cat_cols, folds=5, groups=groups)

    # OOF değerlendirme
    cat_metrics = evaluate_predictions("CatBoost OOF", target, catboost_cv["oof"]) if HAVE_CATBOOST else {"custom_score": 0.0}
    lgb_metrics  = evaluate_predictions("LightGBM OOF", target, lightgbm_cv["oof"])

    # ==== Ensemble ağırlığı grid araması (0..1 step 0.05) ====
    print("Searching blend weight on OOF (custom metric)...")
    best, best_w = -1.0, (0.0, 1.0)
    grid = np.linspace(0, 1, 21)
    for w in grid:
        if HAVE_CATBOOST:
            oof_blend = w * catboost_cv["oof"] + (1 - w) * lightgbm_cv["oof"]
        else:
            oof_blend = lightgbm_cv["oof"]
        mets = evaluate_predictions(f"Blend w={w:.2f}", target, oof_blend)
        if mets["custom_score"] > best:
            best = mets["custom_score"]; best_w = (w, 1 - w)
    print(f"Best blend weights (CAT, LGB): {best_w} | OOF custom={best:.5f}")

    if HAVE_CATBOOST:
        final_test_pred = best_w[0] * catboost_cv["test"] + best_w[1] * lightgbm_cv["test"]
        final_oof = best_w[0] * catboost_cv["oof"] + best_w[1] * lightgbm_cv["oof"]
    else:
        final_test_pred = lightgbm_cv["test"]; final_oof = lightgbm_cv["oof"]

    _ = evaluate_predictions("Final Ensemble OOF", target, final_oof)

    # Submission (orijinal akış korunur)
    submission = pd.read_csv(sample_submission_path)
    submission = submission.drop(columns=[c for c in submission.columns if c != "cust_id"], errors="ignore")
    submission = submission.merge(pd.DataFrame({"cust_id": test_df["cust_id"], "churn": np.clip(final_test_pred, 0.0, 1.0)}), on="cust_id", how="right")
    submission_path = data_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Özet importances
    if HAVE_CATBOOST:
        print("Top CatBoost features:\n", catboost_cv["importance"].head(15))
    print("Top LightGBM features:\n", lightgbm_cv["importance"].head(15))


if __name__ == "__main__":
    main()
