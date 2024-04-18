import os
from itertools import product
from typing import List
import numpy as np
import pandas as pd


def create_empty_df(datasets: List, mask_ratios: List, models: List) -> pd.DataFrame:
    df_results = []

    for dataset in datasets:
        for mask_ratio in mask_ratios:

            row = {
                ("Models", "Dataset"): dataset,
                ("Models", "Mask Ratio"): mask_ratio,
                # (f"{entry['model']}", "MSE"): None,
                # (f"{entry['model']}", "MAE"): None,
            }
            for model in models:
                row[(f"{model}", "MSE")] = None
                row[(f"{model}", "MAE")] = None
            df_results.append(row)
    df = pd.DataFrame(df_results)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index = pd.MultiIndex.from_product([datasets, mask_ratios], names=["Dataset", "Mask Ratio"])
    return df


def create_empty_df_params(datasets: List, models: List) -> pd.DataFrame:
    df_results = []
    excluded = ['Saits', 'TimesNet', 'Transformer']
    filtered_models = [model for model in models if model not in excluded]
    for dataset in datasets:
        row = {}
        for model in filtered_models:
            row[(f"{model}", "all_params")] = None
            row[(f"{model}", "trainable_params")] = None
            row[(f"{model}", "percentage")] = None
        df_results.append(row)
    df_struc = pd.DataFrame(df_results)
    df_struc.columns = pd.MultiIndex.from_tuples(df_struc.columns)
    df_struc = df_struc.set_index([pd.Index(datasets)])
    return df_struc


def generate_all_combinations(models: List, datasets: List, mask_ratios: List) -> List:
    return list(product(models, datasets, mask_ratios))


def get_best_results(experiments: List, folders: List, run_name: str) -> List:
    best_results = []
    for experiment in experiments:
        result_folders = []
        for folder in folders:
            experiment_and_folder_match = False
            i = 0
            for item in experiment:
                if item in folder:
                    i += 1
            if i == 3:
                experiment_and_folder_match = True
            if experiment_and_folder_match:
                # print(folder + ' ' + str(experiment))
                result_folders.append(folder)

        best_mae_value = None
        best_mse_value = None
        best_mae_experiment_folder = None
        best_mse_experiment_folder = None
        trainable_params = None
        all_param = None

        for result_folder in result_folders:
            # metrics
            metrics_path = os.path.join('results/', run_name, result_folder, 'metrics.npy')
            if os.path.exists(metrics_path):
                metrics = np.load(metrics_path)
                # print(metrics_path)
                # print(f"Metrics for {result_folder}: MAE={metrics[0]}, MSE={metrics[1]}")
                mae = metrics[0]
                mse = metrics[1]

                # print(mae)
                # print(best_mae_value)
                if best_mae_value is None or mae < best_mae_value:
                    best_mae_value = mae
                    best_mae_experiment_folder = result_folder
                if best_mse_value is None or mse < best_mse_value:
                    best_mse_value = mse
                    best_mse_experiment_folder = result_folder
            # trainable params
            if experiment[1] in ['Weather', 'Electricity']:
                dataset_name = 'custom'
            else:
                dataset_name = experiment[1]
            params_path = os.path.join('results/', run_name, result_folder,
                                       experiment[0] + 'Imputer_' + dataset_name + '_' + experiment[
                                           2] + '_trainable_params.csv')
            if trainable_params is None:
                if os.path.exists(params_path):
                    params = pd.read_csv(params_path)
                    trainable_params = params.trainable_params[0]
                    all_param = params.all_param[0]
        # print(experiment, best_mse_value, best_mse_experiment_folder)
        best_results.append({
            "experiment": experiment,
            "model": experiment[0],
            "dataset": experiment[1],
            "mask_ratio": experiment[2],
            "results": {
                "best_mae_value": best_mae_value,
                "best_mae_experiment_folder": best_mae_experiment_folder,
                "best_mse_value": best_mse_value,
                "best_mse_experiment_folder": best_mse_experiment_folder,
            },
            "param_config": {
                "trainable_params": trainable_params,
                "all_params": all_param,
                "percentage": round((trainable_params / all_param) * 100,
                                    2) if trainable_params and all_param is not None else 0.000
            }
        })
    return best_results


def get_best_results_comp_tun(experiments: List, folders: List, run_name: str) -> List:
    results = []
    for experiment in experiments:
        result_folders = []
        if experiment[2] == "0125":
            ratio = "0.125"
        elif experiment[2] == "025":
            ratio = "0.25"
        elif experiment[2] == "0375":
            ratio = "0.375"
        elif experiment[2] == "05":
            ratio = "0.5"
        for folder in folders:
            if experiment[1] + '_' + experiment[2] + '_1703_' + experiment[0] + '_iteration_0' in folder:
                experiment_and_folder_match = False
                result_folders.append(folder)
        for result_folder in result_folders:
            mae_value = None
            mse_value = None
            mae_experiment_folder = None
            mse_experiment_folder = None
            trainable_params = None
            all_param = None

            metrics_path = os.path.join('results/', run_name, result_folder, 'metrics.npy')
            if os.path.exists(metrics_path):
                metrics = np.load(metrics_path)
                mae = metrics[0]
                mse = metrics[1]

                mae_value = mae
                mae_experiment_folder = folder
                mse_value = mse
                mse_experiment_folder = folder

                # trainable params
            if experiment[1] in ['Weather', 'Electricity']:
                dataset_name = 'custom'
            else:
                dataset_name = experiment[1]

            params_path = os.path.join('results/', run_name, result_folder,
                                       'GptImputer_' + dataset_name + '_' + ratio + '_trainable_params.csv')
            if trainable_params is None:
                if os.path.exists(params_path):
                    params = pd.read_csv(params_path)
                    trainable_params = params.trainable_params[0]
                    all_param = params.all_param[0]

            results.append({
                "experiment": experiment,
                "model": experiment[0],
                "dataset": experiment[1],
                "mask_ratio": ratio,
                "results": {
                    "mae_value": mae_value,
                    "mae_experiment_folder": mae_experiment_folder,
                    "mse_value": mse_value,
                    "mse_experiment_folder": mse_experiment_folder,
                },
                "param_config": {
                    "trainable_params": trainable_params,
                    "all_params": all_param,
                    "percentage": round((trainable_params / all_param) * 100,
                                        2) if trainable_params and all_param is not None else 0.000
                }
            })
    return results


def get_peft_results(folders: List, experiments: List, peft_method: str) -> List:
    results = []
    for experiment in experiments:
        result_folders = []

        if experiment[2] == "0125":
            ratio = "0.125"
        elif experiment[2] == "025":
            ratio = "0.25"
        elif experiment[2] == "0375":
            ratio = "0.375"
        elif experiment[2] == "05":
            ratio = "0.5"

        for folder in folders:
            if experiment[1] + '_' + experiment[2] + '_2703_' + experiment[0] in folder:
                experiment_and_folder_match = False
                result_folders.append(folder)

        for result_folder in result_folders:
            mae_value = None
            mse_value = None
            mae_experiment_folder = None
            mse_experiment_folder = None
            trainable_params = None
            all_param = None

            metrics_path = os.path.join('results/peft_methods_tuning/', peft_method, result_folder, 'metrics.npy')

            if os.path.exists(metrics_path):
                metrics = np.load(metrics_path)
                mae = metrics[0]
                mse = metrics[1]

                mae_value = mae
                mae_experiment_folder = folder
                mse_value = mse
                mse_experiment_folder = folder

                # trainable params
            if experiment[1] in ['Weather', 'Electricity']:
                dataset_name = 'custom'
            else:
                dataset_name = experiment[1]

            params_path = os.path.join('results/peft_methods_tuning/', peft_method, result_folder,
                                       'GptImputer_' + dataset_name + '_' + ratio + '_trainable_params.csv')
            if trainable_params is None:
                if os.path.exists(params_path):
                    params = pd.read_csv(params_path)
                    trainable_params = params.trainable_params[0]
                    all_param = params.all_param[0]

            results.append({
                "experiment": experiment,
                "model": experiment[0],
                "dataset": experiment[1],
                "mask_ratio": ratio,
                "results": {
                    "mae_value": mae_value,
                    "mae_experiment_folder": mae_experiment_folder,
                    "mse_value": mse_value,
                    "mse_experiment_folder": mse_experiment_folder,
                },
                "param_config": {
                    "trainable_params": trainable_params,
                    "all_params": all_param,
                    "percentage": round((trainable_params / all_param) * 100,
                                        2) if trainable_params and all_param is not None else 0.000
                }
            })

    return results


def evaluate_run(run_name: str, experiments: List, datasets: List, mask_ratios: List, models: List) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    empty_df = create_empty_df(datasets=datasets, mask_ratios=mask_ratios, models=models)
    empty_df_params = create_empty_df_params(datasets=datasets, models=models)

    results_directory = f'./results/{run_name}/'
    folders = os.listdir(results_directory)

    best_results = get_best_results(experiments=experiments, folders=folders, run_name=run_name)

    for entry in best_results:
        index_specified_exp = empty_df[(empty_df[("Models", "Dataset")] == entry["dataset"]) & (
                empty_df[("Models", "Mask Ratio")] == entry["mask_ratio"])]
        empty_df.loc[index_specified_exp.index[0], (entry['model'], "MSE")] = float(
            entry['results'].get('best_mse_value')) if entry['results'].get('best_mse_value') is not None else 0.000
        empty_df.loc[index_specified_exp.index[0], (entry['model'], "MAE")] = float(
            entry['results'].get('best_mae_value')) if entry['results'].get('best_mae_value') is not None else 0.000
        # print(index_specified_exp.index[0][0])
        if entry['model'] not in ['Saits', 'TimesNet', 'Transformer']:
            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "all_params")] = float(
                entry['param_config'].get('all_params')) if entry['param_config'].get(
                'all_params') is not None else 0.000
            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "trainable_params")] = float(
                entry['param_config'].get('trainable_params')) if entry['param_config'].get(
                'trainable_params') is not None else 0.000
            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "percentage")] = float(
                entry['param_config'].get('percentage')) if entry['param_config'].get(
                'percentage') is not None else 0.000

    params_df = empty_df_params
    filled_df = empty_df.drop(columns=[("Models", "Dataset"), ("Models", "Mask Ratio")])
    filled_df = filled_df.sort_index()

    filled_df_summary_agg = filled_df.groupby(level=0).mean()

    return filled_df, filled_df_summary_agg, params_df


def evaluate_components_tuning_run(empty_df: pd.DataFrame, empty_df_params: pd.DataFrame, results) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    for entry in results:
        index_specified_exp = empty_df[(empty_df[("Models", "Dataset")] == entry["dataset"]) & (
                empty_df[("Models", "Mask Ratio")] == entry["mask_ratio"])]
        empty_df.loc[index_specified_exp.index[0], (entry['model'], "MSE")] = float(
            entry['results'].get('mse_value')) if entry['results'].get('mse_value') is not None else 0.000
        empty_df.loc[index_specified_exp.index[0], (entry['model'], "MAE")] = float(
            entry['results'].get('mae_value')) if entry['results'].get('mae_value') is not None else 0.000

        empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "all_params")] = float(
            entry['param_config'].get('all_params')) if entry['param_config'].get('all_params') is not None else 0.000
        empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "trainable_params")] = float(
            entry['param_config'].get('trainable_params')) if entry['param_config'].get(
            'trainable_params') is not None else 0.000
        empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "percentage")] = float(
            entry['param_config'].get('percentage')) if entry['param_config'].get('percentage') is not None else 0.000

    params_df = empty_df_params
    filled_df = empty_df.drop(columns=[("Models", "Dataset"), ("Models", "Mask Ratio")])
    filled_df = filled_df.sort_index()

    filled_df_summary_agg = filled_df.groupby(level=0).mean()

    return filled_df, filled_df_summary_agg, params_df


def evaluate_peft_run(experiments: List, datasets: List, mask_ratios: List, peft_methods: List,
                      mask_ratios_init: List) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    empty_df = create_empty_df(datasets=datasets, mask_ratios=mask_ratios_init, models=peft_methods)
    empty_df_params = create_empty_df_params(datasets=datasets, models=peft_methods)
    for elem in peft_methods:
        results_directory = f'./results/peft_methods_tuning/{elem}/'
        folders = os.listdir(results_directory)

        experiments_peft = generate_all_combinations(datasets=datasets, mask_ratios=mask_ratios, models=peft_methods)
        results = get_peft_results(folders=folders, experiments=experiments_peft, peft_method=elem)

        for entry in results:
            index_specified_exp = empty_df[(empty_df[("Models", "Dataset")] == entry["dataset"]) & (
                    empty_df[("Models", "Mask Ratio")] == entry["mask_ratio"])]
            empty_df.loc[index_specified_exp.index[0], (entry['model'], "MSE")] = float(
                entry['results'].get('mse_value')) if entry['results'].get('mse_value') is not None else 0.000
            empty_df.loc[index_specified_exp.index[0], (entry['model'], "MAE")] = float(
                entry['results'].get('mae_value')) if entry['results'].get('mae_value') is not None else 0.000

            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "all_params")] = float(
                entry['param_config'].get('all_params')) if entry['param_config'].get(
                'all_params') is not None else 0.000
            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "trainable_params")] = float(
                entry['param_config'].get('trainable_params')) if entry['param_config'].get(
                'trainable_params') is not None else 0.000
            empty_df_params.loc[index_specified_exp.index[0][0], (entry['model'], "percentage")] = float(
                entry['param_config'].get('percentage')) if entry['param_config'].get(
                'percentage') is not None else 0.000

    params_df = empty_df_params
    filled_df = empty_df.drop(columns=[("Models", "Dataset"), ("Models", "Mask Ratio")])
    filled_df = filled_df.sort_index()
    filled_df_summary_agg = filled_df.groupby(level=0).mean()

    return filled_df, filled_df_summary_agg, params_df


def prepare_params_table_export(index: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    parameters = parameters[:1]
    df = pd.DataFrame(columns=["all_params", "trainable_params", "percentage"], index=index)

    for col in parameters.columns:
        model = col[0]
        column_name = col[1]
        df[column_name][model] = parameters[(model, column_name)]["ETTh1"]

    return df
