import os
import time

import pandas as pd


def get_result_df(DIR="language_confusion/results/random_forest", level="line_level", mode="mono"):
    regressor = []
    mse = []

    for file in os.listdir(f"{DIR}/{level}/{mode}"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(f"{DIR}/{level}/{mode}", file), index_col=0)
            filename = file.replace(".csv", "")
            mean_mse = df.at["avg", "mse"]
            mse.append(mean_mse)
            regressor.append(filename)

    df = pd.DataFrame.from_dict({
        "variables": regressor,
        "mse": mse})
    df = df.sort_values(by=["mse"])

    return df


def collect_results(output_dir="language_confusion/results/random_forest"):
    for level in ["line_level", "word_level"]:

        for mode in ["mono", "multi", "mono+multi"]:
            outputfile = f"{output_dir}/resutls_{level}_{mode}.csv"
            df = get_result_df(output_dir, level, mode)
            df.to_csv(outputfile, index=False)


def get_avg_mse_regression_into_one(DIR = "language_confusion/results/random_forest"):
    df_list =[]
    for file in os.listdir(DIR):
        filepath = os.path.join(DIR, file)
        if file.endswith(".csv"):
            model_type = file.replace(".csv", "").replace("resutls_", "")
            df = pd.read_csv(filepath, index_col=0)
            df = df.rename(columns={"mse": model_type})
            df_list.append(df)
    df_all = pd.concat(df_list, axis=1, join="inner")
    print(df_all)
    df_all = df_all.sort_values(by=["line_level_mono+multi"])
    df_all.to_csv("language_confusion/results/random_forest_avg.csv")


if __name__ == '__main__':
    collect_results()

    time.sleep(2)

    get_avg_mse_regression_into_one()

