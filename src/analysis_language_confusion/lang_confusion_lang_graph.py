from ast import literal_eval
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances


def calculate_kl_divergence(df1, df2):
    # Flatten the DataFrames to 1D arrays after sorting rows and columns
    flattened_df1 = df1.sort_index(axis=0).sort_index(axis=1).values.flatten().reshape(1, -1)
    flattened_df2 = df2.sort_index(axis=0).sort_index(axis=1).values.flatten().reshape(1, -1)

    # Calculate the KL divergence between the two flattened arrays
    kl_divergence = nan_euclidean_distances(flattened_df1, flattened_df2)

    return kl_divergence


def get_reweighted_entropy(eval_lang,pred_langs_dist):
    "Re-weighted entropy"
    reweighted_entropy = 0
    for lang, prob in pred_langs_dist.items():
        if lang == eval_lang:
            reweighted_entropy -= (1 - prob) * np.log(prob)
        else:
            reweighted_entropy -= prob * np.log(prob)
    return reweighted_entropy


def get_norm_entropy(arr):
    total = sum(arr)
    probabilities = [x / total for x in arr]
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)  # Avoid log(0) errors
    return entropy


def align_dataframes(df1, df2, fill_value=np.nan):
    # Get the union of the row and column indices from both DataFrames

    common_rows = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)
    print(common_rows)
    print(common_columns)

    # Reindex both DataFrames to have only the common rows and columns
    df1_aligned = df1.reindex(index=common_rows, columns=common_columns)
    df2_aligned = df2.reindex(index=common_rows, columns=common_columns)

    return df1_aligned, df2_aligned


def kl_divergence(P, Q):
    # Ensure both matrices are normalized as probability distributions
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Add a small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    P = P + epsilon
    Q = Q + epsilon

    print(P)
    print(Q)

    # Compute the element-wise KL divergence
    kl_div = np.sum(P * np.log(P / Q))

    return kl_div

def get_divergence_score(filepath="results/prompting_language_confusion/dataframes/all_weighted_entropy.csv", level="line"):
    df = pd.read_csv(filepath)
    df = df[df["task"] == "crosslingual"]

    df[f"{level}_level_pred_langs_dist"] = df[f"{level}_level_pred_langs_dist"].apply(literal_eval)

    lang2lang_dist = defaultdict(dict)
    for lang, pred_langs in zip(df["lang"], df[f"{level}_level_pred_langs_dist"]):
        for pred_lang, v in pred_langs.items():
            if pred_lang not in lang2lang_dist[lang]:
                    lang2lang_dist[lang][pred_lang] = list()
            lang2lang_dist[lang][pred_lang].append(v)

    df_lang2lang = pd.DataFrame(lang2lang_dist)
    df_lang2lang = df_lang2lang.dropna()
    df_lang2lang = df_lang2lang.applymap(get_norm_entropy)
    df_lang2lang = df_lang2lang.dropna()

    # dataframe entropy
    df_gb = pd.read_csv("datasets/lang2lang/grambank_sim.csv",index_col=0)
    df_wals = pd.read_csv("datasets/lang2lang/wals_sim.csv", index_col=0)

    df_lang = pd.read_csv("datasets/languages/languoid.csv")
    id2iso = dict(zip(df_lang["id"], df_lang["iso639P3code"]))
    id2name = dict(zip(df_lang["id"], df_lang["name"]))

    def processing_df(df):
        df.index = df.index.map(id2iso)
        df.columns = df.columns.map(id2iso)

        # Drop duplicated columns in the dataframe
        df = df.loc[:, ~df.columns.duplicated()]

        # Drop duplicated rows
        df = df.drop_duplicates()

        # Optionally reset index if needed (you may skip this if not necessary)
        if df.index.duplicated().any():
            df = df.reset_index()

        # Drop duplicated index values, keeping the first occurrence
        df = df[~df.index.duplicated(keep='first')]

        # Set the first column as the new index (conditional)
        df = df.set_index(df.columns[0])

        # Drop duplicated columns (transposed)
        df = df.T.drop_duplicates().T

        # Drop rows where the index is NaN
        df = df[~df.index.isna()]

        return df

    df_gb = processing_df(df_gb)
    df_wals = processing_df(df_wals)

    df_entropy, df_gb_ = align_dataframes(df_lang2lang, df_gb, fill_value=0)
    df_entropy_wals, df_wals_ = align_dataframes(df_lang2lang, df_wals, fill_value=0)
    print(df_entropy.shape, df_entropy_wals.shape)
    print(kl_divergence(df_entropy.to_numpy(), df_gb_.to_numpy()))  # 1.23
    print(kl_divergence(df_entropy_wals.to_numpy(), df_wals_.to_numpy()))  # 1.6086


if __name__ == '__main__':
    get_divergence_score()