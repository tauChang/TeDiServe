import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from typing import Optional, Union

def evaluate_only(model_path: str,
                  features_path: str,
                  test_path: Union[str, list[str]]) -> None:
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    with open(features_path, "r") as f:
        included_features = [line.strip() for line in f.readlines()]
    
    if type(test_path) == str:
        test_path = [test_path]
    
    df = pd.concat([load_and_transform(p) for p in test_path], ignore_index=True)
    df["id"] =df["id"]

    # only keep id with cmpl-0-0
    # df = df[df["id"].str.contains("cmpl-0-0")]
    
    X_test  = df[included_features]
    y_test  = df["denoise_ratio"]
    
    y_pred = model.predict(X_test)
    
    rmse_ratio = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE (ratio): {rmse_ratio:.4f}")
    # WMAPE
    # df = df[included_features]
    results_df = df.copy()
    results_df["y_pred_ratio"] = y_pred
    results_df["y_test_ratio"] = y_test
    results_df["y_pred_steps"] = (y_pred * results_df["output_length"])
    results_df["y_test_steps"] = (y_test * results_df["output_length"])
    numerator = np.abs(results_df["y_test_steps"] - results_df["y_pred_steps"]).sum()
    denominator = np.abs(results_df["y_test_steps"]).sum()
    wmape = numerator / denominator
    print(f"WMAPE (steps): {wmape:.4f}")

    # R^2
    r2_ratio  = r2_score(results_df["y_test_ratio"], results_df["y_pred_ratio"])
    print(f"R^2 (ratio): {r2_ratio:.4f}")
    r2_steps  = r2_score(results_df["y_test_steps"], results_df["y_pred_steps"])
    print(f"R^2 (steps): {r2_steps:.4f}")

    # plot
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Denoise Ratio")
    plt.ylabel("Predicted Denoise Ratio")
    plt.title(f"Predicted vs Actual Denoise Ratio")
    # plot y = x
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig("denoise_ratio_prediction_evaluate_only.png", dpi=300)

    results_df = results_df.round(6)
    results_df.to_csv("denoise_step_prediction_results.csv", index=False)
    

def plot_final_denoise_histogram(path: list[str]) -> None:
    """Plot histogram of final denoise steps per sequence."""
    df = pd.concat([pd.read_json(p, lines=True) for p in path], ignore_index=True)
    final_denoise = df.groupby("id")["num_denoise_ran"].max()
    final_denoise_ratio = final_denoise / df.groupby("id")["output_length"].first()
    print(f"average final denoise ratio: {final_denoise_ratio.mean():.4f}")
    plt.figure(figsize=(6,4))
    plt.hist(final_denoise_ratio, bins=30, edgecolor='black', alpha=
                0.7)
    # x range 0 to 1
    plt.xlim(0, 1)
    plt.xlabel("# Denoise Steps / Output Length")
    plt.ylabel("Count")
    plt.title("GSM8K")
    plt.tight_layout()
    plt.savefig("final_denoise_histogram.png", dpi=300)
    
import pandas as pd
import numpy as np
from collections import defaultdict

def compute_masked_confidence(df: pd.DataFrame):
    # store results in parallel to df rows
    masked_conf_avgs = []

    # group by request id
    for rid, group in df.groupby("id"):
        # sort by denoise step
        group = group.sort_values("num_denoise_ran")

        # track cumulative unmasked tokens for this request
        cumulative_unmasked = set()

        for _, row in group.iterrows():
            output_len = row["output_length"]
            conf = row["confidences"][-output_len:]  # non-prompt confidences
            prompt_len = len(row["confidences"]) - output_len

            # update cumulative unmask set
            newly = row["cur_tokens_unmasked"]
            cumulative_unmasked |= set(newly)

            # compute masked positions
            masked_positions = [i for i in range(output_len) if prompt_len+i not in cumulative_unmasked]
            # print(masked_positions)

            # compute masked confidence average
            if len(masked_positions) == 0:
                # masked_conf_avgs.append(np.nan)
                masked_conf_avgs.append(1)
            else:
                vals = [conf[i] for i in masked_positions]
                # vals = conf
                # print(len(vals))
                # print(vals)
                # masked_conf_avgs.append(float(np.mean(vals)))
                masked_conf_avgs.append(float(np.mean(vals)))
            # print(masked_conf_avgs[-1])
        # 3/0

    df["masked_conf_avg"] = masked_conf_avgs
    return df


def compute_output_confidence(df: pd.DataFrame):
    # store results in parallel to df rows
    masked_conf_avgs = []

    # group by request id
    for rid, group in df.groupby("id"):
        # sort by denoise step
        group = group.sort_values("num_denoise_ran")

        # track cumulative unmasked tokens for this request
        cumulative_unmasked = set()

        for _, row in group.iterrows():
            output_len = row["output_length"]
            conf = row["confidences"][-output_len:]  # non-prompt confidences
            # conf = row["confidences"]
            # conf = row["confidences"]
            prompt_len = len(row["confidences"]) - output_len

            # update cumulative unmask set
            newly = row["cur_tokens_unmasked"]
            cumulative_unmasked |= set(newly)

            # print(vals)
            masked_conf_avgs.append(float(np.mean(conf)))
            # print(masked_conf_avgs[-1])
        # 3/0

    df["output_conf_avg"] = masked_conf_avgs
    return df

def compute_all_confidence_stats(df: pd.DataFrame):
    min_confidences = []
    max_confidences = []
    lower_quartile_confidences = []
    upper_quartile_confidences = []
    mean_confidences = []
    median_confidences = []
    
    for _, row in df.iterrows():
        output_len = row["output_length"]
        conf = row["confidences"][-output_len:]  # non-prompt confidences
        
        min_confidences.append(float(np.min(conf)))
        max_confidences.append(float(np.max(conf)))
        lower_quartile_confidences.append(float(np.percentile(conf, 25)))
        upper_quartile_confidences.append(float(np.percentile(conf, 75)))
        mean_confidences.append(float(np.mean(conf)))
        median_confidences.append(float(np.median(conf)))
    df["min_all_confidence"] = min_confidences
    df["max_all_confidence"] = max_confidences
    df["lower_quartile_all_confidence"] = lower_quartile_confidences
    df["upper_quartile_all_confidence"] = upper_quartile_confidences
    df["mean_all_confidence"] = mean_confidences
    df["median_all_confidence"] = median_confidences
    return df

def load_and_transform(path: str) -> pd.DataFrame:
    """Load a stats.json file and compute derived fields."""
    df = pd.read_json(path, lines=True)
    df["source_file"] = path
    df["full_progress"] = (df["num_denoise_ran"]+1) / df["output_length"]
    df["full_unmask_progress"] = df["num_unmasked_tokens"] / df["output_length"]
    df["block_progress"] = df["block"] / (df["output_length"] / df["block_size"])
    df["block_unmask_progress"] = df["block_num_unmasked_tokens"] / df["block_size"]
    final_denoise = df.groupby("id")["num_denoise_ran"].transform("max")
    df["denoise_ratio"] = (final_denoise - df["num_denoise_ran"]) / df["output_length"]
    df["num_denoise_left"] = final_denoise - df["num_denoise_ran"]
    df = df.sort_values(["id", "num_denoise_ran"]).reset_index(drop=True)
    # Difference in unmasked tokens between consecutive denoise steps
    df["num_cur_unmasked_tokens"] = df.groupby("id")["num_unmasked_tokens"].diff().fillna(df["num_unmasked_tokens"])
    df["full_cur_unmask_progress"] = df["num_cur_unmasked_tokens"] / df["output_length"]

    # compute_masked_confidence(df)
    # compute_output_confidence(df)
    # compute_all_confidence_stats(df)
    df["last_recompute_avg_output_confidence"] = df["avg_confidence"]
    df["cur_avg_output_confidence"] = df["avg_confidence"]

    return df[[
        "source_file",
        "id",
        "confidence_threshold",
        "num_unmasked_tokens",
        "num_cur_unmasked_tokens",
        "output_length",
        "block_size",
        "block",
        "block_num_unmasked_tokens",
        "num_denoise_ran",
        "num_denoise_left",
        "full_progress",
        "full_unmask_progress",
        "full_cur_unmask_progress",
        "block_progress",
        "block_unmask_progress",
        "denoise_ratio",
        # "masked_conf_avg",
        # "output_conf_avg",
        "last_recompute_avg_output_confidence",
        "cur_avg_output_confidence",
        # "min_all_confidence",
        # "max_all_confidence",
        # "lower_quartile_all_confidence",
        # "upper_quartile_all_confidence",
        # "mean_all_confidence",
        # "median_all_confidence",
        # "min_confidence",
        # "q25_confidence",
        # "median_confidence",
        # "avg_confidence",
        # "q75_confidence",
        "output_min_confidence",
        "output_q25_confidence",
        "output_median_confidence",
        "output_avg_confidence",
        "output_q75_confidence",
    ]]
    
def mape_objective(y_pred, dataset):
    """Custom MAPE objective for LightGBM."""
    y_true = dataset.get_label()
    eps = 1e-6  # avoid div by zero
    grad = np.sign(y_pred - y_true) / (np.abs(y_true) + eps)
    hess = np.ones_like(y_true) / (np.abs(y_true) + eps)
    return grad, hess

def mape_metric(y_pred, dataset):
    """MAPE metric for LightGBM."""
    y_true = dataset.get_label()
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps)))
    return "mape", mape, False  # lower is better


def train_and_evaluate(train_path: Union[str, list[str]],
                       model_name: str = "tmp",
                       test_path: Optional[Union[str, list[str]]] = None) -> None:
    # objective = "num_denoise_left"
    objective = "denoise_ratio"
    included_features = [
        # "id",
        # "num_denoise_ran",
        # "num_denoise_left",

        # "confidence_threshold",
        # "num_unmasked_tokens",
        # "output_length",
        # "block_size",
        # "block",
        # "block_num_unmasked_tokens",
        # "block_progress",
        # "block_unmask_progress",
        # "full_cur_unmask_progress",
        # "full_progress",
        # "full_unmask_progress",

        "full_progress",
        "full_unmask_progress",
        "confidence_threshold",
        "output_min_confidence",
        "output_q25_confidence",
        "output_median_confidence",
        "output_avg_confidence",
        "output_q75_confidence",

        # "masked_conf_avg",
        # "output_conf_avg",
        # "last_recompute_avg_output_confidence",
        # "cur_avg_output_confidence",
        # "denoise_ratio"
        # "num_cur_unmasked_tokens",
        # "min_all_confidence",
        # "max_all_confidence",
        # "lower_quartile_all_confidence",
        # "upper_quartile_all_confidence",
        # "mean_all_confidence",
        # "median_all_confidence",
        # "min_confidence",
        # "q25_confidence",
        # "median_confidence",
        # "avg_confidence",
        # "q75_confidence",
    ]

    if type(train_path) == str:
        train_path = [train_path]
    if test_path is not None and type(test_path) == str:
        test_path = [test_path]

    plot_final_denoise_histogram(train_path)
    df = pd.concat([load_and_transform(p) \
        for p in [*train_path, *(test_path if test_path else [])]], 
                   ignore_index=True)
    df["id"] = df["source_file"] + "_" + df["id"].astype(str)
    print(df)
    if test_path is None:
        unique_ids = df["id"].unique()
        train_ids, test_ids = train_test_split(
            unique_ids, test_size=0.2, random_state=42
        )
        title_suffix = " (80/20 split)"
    else:
        # train_ids = pd.concat([pd.read_json(p, lines=True)["id"] for p in train_path], ignore_index=True).unique()
        # test_ids  = pd.concat([pd.read_json(p, lines=True)["id"] for p in test_path], ignore_index=True).unique()
        train_ids = df[df["source_file"].isin(train_path)]["id"].unique()
        test_ids  = df[df["source_file"].isin(test_path)]["id"].unique()
        title_suffix = f" (Train on {', '.join(train_path)}, Test on {', '.join(test_path)})"
    
    train_set = df[df["id"].isin(train_ids)]
    test_set  = df[df["id"].isin(test_ids)]
    X_train = train_set.drop(columns=[objective])[included_features]
    X_test  = test_set.drop(columns=[objective])[included_features]
    y_train = train_set[objective]
    y_test  = test_set[objective]
    
    print(f"X_train: {X_train}")
    print(f"X_test: {X_test}")
    # Train model

    # use a model that just predicts the mean based on confidence threshold
    

    # linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # # # # print parameters
    # print("Model parameters:")
    # for feature, coef in zip(X_train.columns, model.coef_):
    #     print(f"  {feature}: {coef:.16f}")
    # print(f"  Intercept: {model.intercept_:.4f}")

    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # model.fit(X_train, y_train)

    import lightgbm as lgb
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        {
            'objective': 'regression',
            'metric': 'rmse',
            # 'objective': 'quantile',
            # 'alpha': 0.6,  # for quantile regression
            # 'learning_rate': 0.2,
            # 'num_leaves': 800,
            # 'max_depth': 16,
            'verbose': -1
        },
        train_data,
    )
    # model = lgb.train(
    #     {
    #         'objective': mape_objective,
    #         'metric': 'None',
    #         'verbose': -1
    #     },
    #     train_data,
    #     feval=mape_metric
    # )
    # # zip features and importances
    # feature_importances = sorted(zip(X_train.columns, model.feature_importance()), key=lambda x: x[1], reverse=True)
    # print("Feature importances:")
    # for feature, importance in feature_importances:
    #     print(f"  {feature}: {importance}")
    # # model.save_model("denoise_ratio_model.txt")

    # import xgboost as xgb
    # model = xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     # objective='reg:quantileerror',
    #     # alpha=0.5,  # for quantile regression
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     max_depth=6,
    #     n_jobs=-1,
    #     verbosity=0
    # )
    # model.fit(X_train, y_train)
    
    

    # save the model and features used to ./models
    # name = "dual_cache_32_1120"
    # name = "dual_cache_1120_256_32"
    # name = "dual_cache_1120_256_32_tiny"
    # name = "tmp_quantile"
    model_path = f"models/lgb/{model_name}/model.bin"
    features_path = f"models/lgb/{model_name}/features.txt"
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    if hasattr(model, "save_model"):
        model.save_model(model_path)
    with open(features_path, "w") as f:
        for feature in included_features:
            f.write(f"{feature}\n")
    
    # # Evaluate
    import time
    # predict one by one 
    pred_time = []
    for i in range(min(500, len(X_test))):
        start_time = time.time()
        _ = model.predict(X_test.iloc[i:i+10])
        pred_time.append(time.time() - start_time)
    print(f"Average prediction time per sample (single): {sum(pred_time) / len(pred_time):.6f} seconds")
    
    y_pred = model.predict(X_test)
    # convert to actual step
    # y_pred *= y_test_output_length
    # y_test *= y_test_output_length
    results_df = test_set.copy()
    if objective == "denoise_ratio":
        results_df["y_pred_ratio"] = y_pred
        results_df["y_test_ratio"] = y_test
        results_df["y_pred_steps"] = (y_pred * test_set["output_length"])
        results_df["y_test_steps"] = (y_test * test_set["output_length"])
    elif objective == "num_denoise_left":
        results_df["y_pred_steps"] = y_pred
        results_df["y_test_steps"] = y_test
        results_df["y_pred_ratio"] = (y_pred / test_set["output_length"])
        results_df["y_test_ratio"] = (y_test / test_set["output_length"])
    
    print(results_df.head())

    rmse_steps = root_mean_squared_error(results_df["y_test_steps"], results_df["y_pred_steps"])
    rmse_ratio = root_mean_squared_error(results_df["y_test_ratio"], results_df["y_pred_ratio"])
    # absolute percentage error of step
    # exclude those with y_test_steps = 0
    # results_df = results_df[results_df["y_test_steps"] > 0]
    # abs_percentage_error = np.abs(results_df["y_test_steps"] - results_df["y_pred_steps"]) / results_df["y_test_steps"]
    # mean_abs_percentage_error = abs_percentage_error.mean()
    # print(f"Mean Absolute Percentage Error (steps): {mean_abs_percentage_error:.4f}")
    numerator = np.abs(results_df["y_test_steps"] - results_df["y_pred_steps"]).sum()
    denominator = np.abs(results_df["y_test_steps"]).sum()
    wmape = numerator / denominator
    print(f"WMAPE (steps): {wmape:.4f}")
    print(f"RMSE (ratio): {rmse_ratio:.4f}")
    print(f"RMSE (steps): {rmse_steps:.4f}")
    r2_steps  = r2_score(results_df["y_test_steps"], results_df["y_pred_steps"])
    r2_ratio  = r2_score(results_df["y_test_ratio"], results_df["y_pred_ratio"])
    print(f"R^2 (ratio): {r2_ratio:.4f}")
    print(f"R^2 (steps): {r2_steps:.4f}")

    # QCE (Quantile Calibration Error)
    quantile = 0.8
    # quantile condition: y_true <= y_pred
    coverage = np.mean(results_df["y_test_steps"] <= results_df["y_pred_steps"])
    print(f"Coverage (steps): {coverage:.4f}")
    qce = abs(coverage - quantile)
    print(f"QCE (steps): {qce:.4f}")

    

    # plot error vs. full_unmask_progress
    results_df["full_unmask_progress_rounded"] = results_df["full_unmask_progress"].round(1)
    results_df["masked_tokens_left"] = results_df["output_length"] - results_df["num_unmasked_tokens"]
    results_df["error"] = (results_df["y_pred_steps"] - results_df["y_test_steps"]) / results_df["y_test_steps"] 
    # results_df["error"] = (results_df["y_test_steps"] - results_df["y_pred_steps"])
    plt.figure(figsize=(10,5))
    results_df.boxplot(column="error", by="full_unmask_progress_rounded",
                          grid=False, showfliers=False)
    # write result_df to file
    # drop source_file column
    results_df = results_df.drop(columns=["source_file", "full_unmask_progress_rounded", "error"])
    # print to two decimal places
    # results_df = results_df.round(2)
    results_df.to_csv("denoise_step_prediction_results.csv", index=False)
    
    # y limit -1 and 1
    plt.ylim(-1, 1)
    plt.xlabel("Full Denoise Progress (rounded)")
    plt.ylabel("Prediction Error")
    plt.title(f"Prediction Error vs Full Denoise Progress{title_suffix}")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("denoise_ratio_error_vs_progress.png", dpi=300)
    
    
    # plot num_unmasked_token vs. predicted
    plt.figure(figsize=(6,5))
    plt.scatter(results_df["num_unmasked_tokens"], 
                results_df["y_test_steps"], 
                alpha=0.3,
                label="Actual"
                )
    plt.scatter(results_df["num_unmasked_tokens"],
                results_df["y_pred_steps"],
                alpha=0.3,
                label="Predicted"
    )
    plt.xlabel("Num Unmasked Tokens")
    plt.ylabel("Num Denoise Steps Left")
    plt.title(f"Predicted vs Actual Denoise Steps Left{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("denoise_steps_prediction.png", dpi=300)
    

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Denoise Ratio")
    plt.ylabel("Predicted Denoise Ratio")
    for threshold in results_df["confidence_threshold"].unique():
        subset = results_df[results_df["confidence_threshold"] == threshold]
        plt.scatter(subset["y_test_ratio"], subset["y_pred_ratio"], alpha=0.3, label=f"Threshold {threshold}")
    plt.legend(title="Confidence Threshold")
    plt.title(f"Predicted vs Actual Denoise Ratio{title_suffix}")
    # plot y = x
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig("denoise_ratio_prediction.png", dpi=300)

    

# --- Usage ---
# 1) Single file, internal split
# train_and_evaluate("GSAI-ML_LLaDA-8B-Base_prefix_step_estimator.json")
# train_and_evaluate("../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_128_step_estimator.json")
# train_and_evaluate("../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block32_conf0.8_out256.json")
# evaluate_only(
#     # "models/lgb/dual_cache_32_1120/model.bin",
#     # "models/lgb/dual_cache_32_1120/features.txt",
#     # "models/lgb/tmp/model.bin",
#     # "models/lgb/tmp/features.txt",
#     "models/lgb/dual_cache_256_32/model.bin",
#     "models/lgb/dual_cache_256_32/features.txt",
#     [
#         # "/u/tchang85/dllm/experiment_dir/20251120_164011/step_data.json",
#         # "/u/tchang85/dllm/experiment_dir/20251120_165845/step_data.json"
#         # "/u/tchang85/dllm/experiment_dir/20251120_165033/step_data.json"
#         "/u/tchang85/dllm/current_experiment/step_data.json"
#     ]
# )
# 3/0
cross_test = False
train_and_evaluate(
    # "GSAI-ML_LLaDA-8B-Base_prefix_step_estimator.json",
    # model_name="1121_dual_256_512_1024_quantile_0.6",
    model_name="tmp",
    train_path=[
        # "/u/tchang85/dllm/experiment_dir/20251120_164011/step_data.json",
        "../../step_data_dual_cache_with_output/dual_0.9_256.json",
        "../../step_data_dual_cache_with_output/dual_0.8_256.json",
        "../../step_data_dual_cache_with_output/dual_0.7_256.json",
        "../../step_data_dual_cache_with_output/dual_0.6_256.json",
        "../../step_data_dual_cache_with_output/dual_0.5_256.json",
        "../../step_data_dual_cache_with_output/dual_0.9_512.json",
        "../../step_data_dual_cache_with_output/dual_0.8_512.json",
        "../../step_data_dual_cache_with_output/dual_0.7_512.json",
        "../../step_data_dual_cache_with_output/dual_0.6_512.json",
        "../../step_data_dual_cache_with_output/dual_0.5_512.json",
        "../../step_data_dual_cache_with_output/dual_0.9_1024.json",
        "../../step_data_dual_cache_with_output/dual_0.8_1024.json",
        "../../step_data_dual_cache_with_output/dual_0.7_1024.json",
        "../../step_data_dual_cache_with_output/dual_0.6_1024.json",
        "../../step_data_dual_cache_with_output/dual_0.5_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.9_256.json",
        # "../../step_pred_data_dual_cache/dual_0.8_256.json",
        # "../../step_pred_data_dual_cache/dual_0.7_256.json",
        # "../../step_pred_data_dual_cache/dual_0.6_256.json",
        # "../../step_pred_data_dual_cache/dual_0.5_256.json",
        # "../../step_pred_data_dual_cache/dual_0.9_512.json",
        # "../../step_pred_data_dual_cache/dual_0.8_512.json",
        # "../../step_pred_data_dual_cache/dual_0.7_512.json",
        # "../../step_pred_data_dual_cache/dual_0.6_512.json",
        # "../../step_pred_data_dual_cache/dual_0.5_512.json",
        # "../../step_pred_data_dual_cache/dual_0.9_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.8_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.7_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.6_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.5_1024.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
        # "../../step_data_1106/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_1106/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.9.json",
        # "../../step_data_1106/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.9.json",
        # "../../step_data_kiet/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_kiet/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
        # "../../step_data_kiet/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
        # "../../step_data_kiet/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
        # "../../step_data_kiet/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.5.json",
        # "../../step_data_kiet/gsm8k_100/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_kiet/gsm8k_100/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
        # "../../step_data_kiet/gsm8k_100/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
        # "../../step_data_kiet/gsm8k_100/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
        # "../../step_data_kiet/gsm8k_100/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.5.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block16_conf0.8_out256.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block16_conf0.8_out512.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block16_conf0.9_out256.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block16_conf0.9_out512.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block32_conf0.8_out256.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block32_conf0.8_out512.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block32_conf0.9_out256.json",
        # "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_block32_conf0.9_out512.json",
    ],
    test_path = [
        # "../../step_data_dual_cache_with_output/dual_0.9_256.json",
        # "../../step_data_dual_cache_with_output/dual_0.8_256.json",
        # "../../step_data_dual_cache_with_output/dual_0.7_256.json",
        # "../../step_data_dual_cache_with_output/dual_0.6_256.json",
        # "../../step_data_dual_cache_with_output/dual_0.5_256.json",
        # "../../step_data_dual_cache_with_output/dual_0.9_512.json",
        # "../../step_data_dual_cache_with_output/dual_0.8_512.json",
        # "../../step_data_dual_cache_with_output/dual_0.7_512.json",
        # "../../step_data_dual_cache_with_output/dual_0.6_512.json",
        # "../../step_data_dual_cache_with_output/dual_0.5_512.json",
        # "../../step_data_dual_cache_with_output/dual_0.9_1024.json",
        # "../../step_data_dual_cache_with_output/dual_0.8_1024.json",
        # "../../step_data_dual_cache_with_output/dual_0.7_1024.json",
        # "../../step_data_dual_cache_with_output/dual_0.6_1024.json",
        # "../../step_data_dual_cache_with_output/dual_0.5_1024.json",
        # "/u/tchang85/dllm/experiment_dir/20251121/092606/step_data.json"
        # "/u/tchang85/dllm/experiment_dir/20251121/092606/step_data.json"
        # "../../step_data_dual_cache_with_output/dual_0.5_256.json",
        # "/u/tchang85/dllm/experiment_dir/20251120_164011/step_data.json",
        # "/u/tchang85/dllm/current_experiment/step_data.json"
        # "../../step_pred_data_dual_cache/dual_0.9_512.json",
        # "../../step_pred_data_dual_cache/dual_0.8_512.json",
        # "../../step_pred_data_dual_cache/dual_0.7_512.json",
        # "../../step_pred_data_dual_cache/dual_0.6_512.json",
        # "../../step_pred_data_dual_cache/dual_0.5_512.json",
        # "../../step_pred_data_dual_cache/dual_0.9_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.8_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.7_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.6_1024.json",
        # "../../step_pred_data_dual_cache/dual_0.5_1024.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/512/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block32_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_block8_conf0.5.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.9.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.8.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.7.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.6.json",
        # "../../step_data_1107_combined/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.5.json",
    ] if cross_test else None
)

# 2) Train on one file, test on another
# train_and_evaluate(
#     # "GSAI-ML_LLaDA-8B-Base_prefix_step_estimator.json",
#     "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_128_step_estimator.json",
#     "../../step_data/GSAI-ML_LLaDA-8B-Base_prefix_suffix_32_step_estimator.json",
# )
