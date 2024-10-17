import pandas as pd
import numpy as np
import json

from ast import literal_eval
from collections import defaultdict
import os


def get_lang2lang_matrix_for_inversion(df, level="line", lingual="crosslingual"):
    if lingual!="all":
        df = df[df["task"] == lingual]

    df = df.dropna(subset=[f"{level}_level_pred_langs_dist"])
    df[f"{level}_level_pred_langs_dist"] = df[f"{level}_level_pred_langs_dist"].apply(literal_eval)

    lang2lang_dist = defaultdict(dict)
    for lang, pred_langs in zip(df["lang"], df[f"{level}_level_pred_langs_dist"]):
        if lang == "ara":
            lang = "arb"
        if lang == "zho":
            lang = "cmn"

        for pred_lang, v in pred_langs.items():
            if pred_lang == "msa":
                pred_lang = "arb"
            if pred_lang == "ara":
                pred_lang = "arb"
            if pred_lang == "zho":
                pred_lang = "cmn"

            if pred_lang not in lang2lang_dist[lang]:
                lang2lang_dist[lang][pred_lang] = list()
            if pred_lang == "eng":
                reweighted = -(1 - v) * np.log(v)
            elif pred_lang == lang:
                reweighted = -(1 - v) * np.log(v)
            else:
                reweighted = -v * np.log(v)

            # maybe change this????
            lang2lang_dist[lang][pred_lang].append(reweighted)

    df_lang2lang = pd.DataFrame(lang2lang_dist)

    df_lang2lang = df_lang2lang.applymap(np.mean)

    df_lang2lang = df_lang2lang.fillna(0)
    df_lang2lang = df_lang2lang.replace({-0: 0})
    # Drop rows where all values are 0
    df_lang2lang = df_lang2lang.loc[~(df_lang2lang == 0).all(axis=1)]

    # Drop columns where all values are 0
    df_lang2lang = df_lang2lang.loc[:, ~(df_lang2lang == 0).all(axis=0)]
    if "unknown" in df_lang2lang.columns:
        df_lang2lang = df_lang2lang.drop(["unknown"])
    if "unk" in df_lang2lang.columns:
        df_lang2lang = df_lang2lang.drop(["unk"])
    return df_lang2lang


def align_dataframes(df1, df2):
    # Get the union of the row and column indices from both DataFrames
    # handle the reindex error,
    df1 = df1[~df1.index.duplicated(keep='first')]
    df2 = df2[~df2.index.duplicated(keep='first')]
    df1 = df1.loc[:, ~df1.columns.duplicated(keep='first')]
    df2 = df2.loc[:, ~df2.columns.duplicated(keep='first')]

    common_rows = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)
    print(common_rows)
    print(common_columns)

    # Reindex both DataFrames to have only the common rows and columns
    df1_aligned = df1.reindex(index=common_rows, columns=common_columns)
    df2_aligned = df2.reindex(index=common_rows, columns=common_columns)

    return df1_aligned, df2_aligned


def process_typ_sim(filepath, df_lang2lang, d_rename={"msa": "arb", "ara": "arb", "zho": "cmn"}):
    df_sim = pd.read_csv(filepath, index_col=0)
    indices = df_lang2lang.index.tolist()
    cols = df_lang2lang.columns.tolist()
    sim_cols = df_sim.columns.tolist()
    sim_indice = df_sim.index.tolist()

    cols = [x for x in cols if x in sim_cols]
    indices = [x for x in indices if x in sim_indice]

    df_sim.rename(columns=d_rename, inplace=True)
    df_sim.rename(index=d_rename, inplace=True)
    df_sim = df_sim[df_sim.index.isin(indices)]
    df_sim = df_sim[cols]

    return df_sim


def kl_divergence_by_cols(df1, df2):
    # Initialize a variable to accumulate the total divergence
    total_kl_divergence = []
    kl_div_dict = {}
    # Loop through each column to compute the KL divergence
    for col in df1.columns:
        # Step 1: Get the values of df1 for the current column, excluding zeros
        df1_col = df1[col]
        df2_col = df2[col]

        # Filter out rows where df1_col is zero to avoid issues with KL divergence
        nonzero_indices = df1_col != 0
        P = df1_col[nonzero_indices].to_numpy()
        Q = df2_col[nonzero_indices].to_numpy()
        # print(P,Q)

        # Step 2: Normalize the values to ensure they are proper probability distributions
        P = P / np.sum(P)
        Q = Q / np.sum(Q)

        # avoid division by zero or log
        epsilon = 1e-10
        P = P + epsilon
        Q = Q + epsilon

        # Step 3: Calculate the KL divergence for the current pair of columns
        kl_div = np.sum(P * np.log(P / Q))
        print(f"KL Divergence for {col}: {kl_div}")
        kl_div_dict[col] = kl_div

        # Accumulate the KL divergence for aggregation
        total_kl_divergence.append(kl_div)
    return np.mean(total_kl_divergence), kl_div_dict


def get_kl_divs(df_l2l, lang_typ_file="datasets/lang2lang/clics3_jaccard.csv",
                outputfolder="results/inversion_language_confusion/kl_divergence", lang_type="ostling"):
    print(f"loading  {lang_typ_file}")
    filename = os.path.basename(lang_typ_file).replace(".csv", "")
    outputfile = os.path.join(outputfolder, filename + ".json")

    df_sim = process_typ_sim(lang_typ_file, df_l2l)
    if lang_type not in ["ostling", "colex2lang"]:
        if "pmi" not in lang_typ_file and "cosine" not in lang_typ_file:
            print("converting the distance to similarity")
            df_sim = 1 - df_sim
    df_sim_aligned, l2l = align_dataframes(df_sim, df_l2l)

    df_sim_aligned.to_csv(os.path.join(outputfolder, f"{filename}_sim_df.csv"))
    l2l.to_csv(os.path.join(outputfolder, f"{filename}_l2l_df.csv"))

    print("calculating KL divergence")
    kl_mean, kl_divs_dict = kl_divergence_by_cols(l2l, df_sim_aligned)
    print(f"{filename} : {kl_mean}")
    with open(outputfile, "w+") as f:
        json.dump(kl_divs_dict, f)


def main():
    filepath = f"results/prompting_language_confusion/dataframes/all_reweighted_entropy.csv"
    print(f"Loading file from {filepath}")

    df = pd.read_csv(filepath)

    for level in ["line"]: # line
        # "crosslingual",
        for lingual in ["crosslingual", "monolingual", "all"]:


            l2l = get_lang2lang_matrix_for_inversion(df, level=level, lingual=lingual)
            outputfolder = f"results/prompting_language_confusion/kl_divergence/{level}/{lingual}/colex2lang"
            if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)

            for file in os.listdir("datasets/colex2lang_sim"):
                if file.endswith(".csv"):
                    lang_typ_file = os.path.join("datasets/colex2lang_sim", file)
                    get_kl_divs(l2l, lang_typ_file, outputfolder, "colex2lang")

            # for file in os.listdir("datasets/ostling_lang2sim"):
            #     if file.endswith(".csv"):
            #         lang_typ_file = os.path.join("datasets/ostling_lang2sim", file)
            #         get_kl_divs(l2l, lang_typ_file, outputfolder, "ostling")


            # for file in os.listdir("datasets/lang2lang"):
            #     if file.endswith(".csv"):
            #         lang_typ_file = os.path.join("datasets/lang2lang", file)
            #         get_kl_divs(l2l, lang_typ_file, outputfolder)
            #
            # for file in os.listdir("datasets/lang2vec_distances"):
            #     if file.endswith(".csv"):
            #         lang_typ_file = os.path.join("datasets/lang2vec_distances", file)
            #         get_kl_divs(l2l, lang_typ_file, outputfolder)


if __name__ == '__main__':
    main()
