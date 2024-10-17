import pandas as pd
import numpy as np
from ast import literal_eval
from collections import defaultdict
import os
from scipy.stats import entropy
from src.analysis_language_confusion.get_stats_corr import get_annotated_corr
from src.analysis_language_confusion.plot_language_confusion_inversion_performance import plot_lc_inversion


# language and their pre-training data percentages in mT5
lang2percent = {'zho': 1.67, 'ara': 1.66, 'eng': 5.67, 'mar': 0.93, 'rus': 3.71, 'tur': 1.93, 'ben': 0.91, 'fas': 1.67,
                'deu': 3.05, 'jpn': 1.92,
                'pol': 2.15, 'spa': 3.09, 'fra': 2.89, 'swe': 1.61, 'kor': 1.14, 'ces': 1.72, 'bul': 1.29, 'ita': 2.43,
                'slk': 1.19, 'aze': 0.82,
                'ukr': 1.51, 'fin': 1.35, 'nld': 1.98, 'tgl': 0.52, 'mkd': 0.62, 'epo': 0.4,
                "pan": 0.37, "sin": 0.41, "guj": 0.43, "urd": 0.61, "ydd": 0.28, "mon": 0.62,
                "amh": 0.29, "kaz": 0.65, "mhr": 0, "arb": 1.66, "heb": 1.06, "cmn": 1.67, "mlt": 0.64, "hun": 1.48,
                "hin": 1.21}

iso2name = {'arb': "Arabic", 'cmn': "Chinese", 'jpn': "Japanese", 'deu': "German", 'tur': "Turkish",
            'guj': "Gujarati", 'heb': "Hebrew", 'hin': "Hindi", 'pan': "Punjabi", 'kaz': "Kazakh", 'mon': "Mongolian",
            'urd': "Urdu", 'amh': "Amharic",
            "hun": "Hungarian", "kor": "Korean", "mhr": "Meadow Mari", "mlt": "Maltese", "sin": "Sinhala",
            "ydd": "Yiddish", "fin": "Finnish"}


def get_inversion_results(metric="bleu_score", lingual="multi"):
    """Get inversion results dataframe."""
    df_results_metric_list = []

    assert lingual in ["multi", "mono"]

    for file in os.listdir("datasets/inversion_language_confusion/inversion_results/"):
        if file.startswith(f"{lingual}lingual_eval_{metric}_") and file.endswith(".csv"):
            step = file.replace(f"multilingual_eval_{metric}_", "").replace(".csv", "").replace("_inverter",
                                                                                                "").replace(
                "_corrector", "")
            step2name = {"base": "Base", "step1": "Step1", "step50_sbeam8": "Step50+sbeam8"}
            print(f"Get result for {metric}-{lingual} from {file}")
            df_ = pd.read_csv(os.path.join("datasets/inversion_language_confusion/inversion_results/", file))
            df_["step"] = step2name[step]

            df_results_metric_list.append(df_)
    df_results_metric = pd.concat(df_results_metric_list)
    df_results_metric.rename(columns={"Unnamed: 0": "model"}, inplace=True)

    def replace_string(x):
        return x.replace("mt5_", "").replace("_32_2layers_inverter", "").replace("_32_2layers_corrector", "")

    df_results_metric["model_name"] = df_results_metric["model"].apply(replace_string)
    df_results_metric["model_name"] = df_results_metric["model_name"].replace({"me5_ara-script": 'me5_arab-script'})

    return df_results_metric


def get_language_confusion_results(lingual="multi", level="line", generation_setting="crosslingual", mode=True):
    assert lingual in ["multi", "mono"]
    assert level in ["line", "word"]
    # in-family, in-script
    train_langs = ['pan', 'heb', 'guj', 'cmn', 'deu', 'kaz', 'urd', 'jpn',
             'tur', 'hin', 'arb', 'mon', 'hin_pan', 'pan_urd', 'kaz_tur',
             'guj_hin', 'deu_tur', 'guj_urd', 'hin_urd', 'arb_heb', 'guj_pan',
             'kaz_mon', 'guj_tur', 'cmn_jpn', 'pan_tur', 'hin_tur', 'kaz_pan']

    df_lc = pd.read_csv(
        f"datasets/inversion_language_confusion/langdist_data_all_langs/dataset2langdist_{level}_level_{lingual}_0.3.csv")

    df_lc = df_lc[df_lc["step"] != "Labels"]
    df_lc["training"] = df_lc["training"].apply(literal_eval)
    df_lc["pred_langs"] = df_lc["pred_langs"].apply(literal_eval)

    # convert the language code to iso3code.
    df_lc["training_langs"] = df_lc["training"].apply(lambda y: [x.split("_")[0] for x in y])

    if mode:
        df_lc["train_langs"]= df_lc["training_langs"].apply(lambda x:"_".join(x))
        df_lc = df_lc[df_lc["train_langs"].isin(train_langs)]

    df_lc["eval_lang"] = df_lc["eval_lang"].apply(lambda x: x.split("_")[0])

    def get_langs_for_row(r):
        l = r["training_langs"] + [r["eval_lang"]]
        return "_".join(l)

    df_lc["train_eval_langs"] = df_lc.apply(lambda row: get_langs_for_row(row), axis=1)
    # make sure that data for each step for the languages is present.
    groups = []
    for k, group in df_lc.groupby(by="train_eval_langs"):
        if len(group) == 3:
            groups.append(group)

    df_lc_groupped = pd.concat(groups, axis=0)

    def filter_lang(r):
        if r["eval_lang"] not in r["training_langs"]:
            return False
        else:
            return True

    df_lc_groupped["eval_in_training"] = df_lc_groupped.apply(lambda row: filter_lang(row), axis=1)

    def prob_dist_norm(x):
        unk_value = round(1 - sum(x.values()), 2)
        if unk_value > 0:
            x["unk"] = unk_value
        return x

    # use the normalized distributions
    df_lc_groupped["pred_langs_dist"] = df_lc_groupped["pred_langs"].apply(prob_dist_norm)

    # get only the relevant columns
    # df_lc_groupped = df_lc_groupped[["model", "training_langs", "eval_lang", "step", "pred_langs_dist"]]
    if generation_setting == "crosslingual":
        return df_lc_groupped[df_lc_groupped["eval_in_training"] == False]

    elif generation_setting == "monolingual":
        return df_lc_groupped[df_lc_groupped["eval_in_training"] == True]

    else:
        return df_lc_groupped


def get_entropy_for_all(x):
    return entropy(np.array(list(x.values())))


def get_entropy_outside_eval_train(row):
    """Get entropy outside eval/train langauges."""
    pred_langs_dist = row["pred_langs_dist"]
    train_langs = row["training_langs"]
    eval_lang = row["eval_lang"]
    dist_others = []
    for lang, dist in pred_langs_dist.items():
        if lang != eval_lang and lang not in train_langs:
            dist_others.append(dist)
    return entropy(np.array(dist_others))


def get_reweighted_entropy(row):
    pred_langs_dist = row["pred_langs_dist"]
    train_langs = row["training_langs"]
    eval_lang = row["eval_lang"]
    reweighted_entropy = 0
    for lang, prob in pred_langs_dist.items():
        if lang in train_langs:
            reweighted_entropy -= (1-prob) * np.log(prob)
        elif lang == eval_lang:
            reweighted_entropy -= (1-prob) * np.log(prob)
        else:
            reweighted_entropy -= prob*np.log(prob)
    return reweighted_entropy


def get_step2eval_lang_entropy_dataframe(df, by_col="weighted_entropy"):
    # average entropies across models for each langauge
    step2eval_lang_entropy = defaultdict(dict)
    step2train_lang_entropy = defaultdict(dict)

    for k, group in df.groupby(by="step"):
        for eval_lang, training_langs, entropy in zip(group["eval_lang"], group["training_langs"],
                                                      group[by_col]):
            if k not in step2eval_lang_entropy[eval_lang]:
                step2eval_lang_entropy[eval_lang][k] = list()
            step2eval_lang_entropy[eval_lang][k].append(entropy)

            # get cmn_jpn: ....
            train_langs = "_".join(sorted(training_langs))
            if k not in step2train_lang_entropy[train_langs]:
                step2train_lang_entropy[train_langs][k] = list()

            step2train_lang_entropy[train_langs][k].append(entropy)
            # for train_lang in training_langs:
            #     if k not in step2train_lang_entropy[train_lang]:
            #         step2train_lang_entropy[train_lang][k] = list()
            #     step2train_lang_entropy[train_lang][k].append(entropy)

    step2eval_lang_entropy_avg = defaultdict(dict)
    step2train_lang_entropy_avg = defaultdict(dict)

    # by eval lang, average the entropies by each language by each step across models.
    for lang, step_dict in step2eval_lang_entropy.items():
        for step, entropy_list in step_dict.items():
            step2eval_lang_entropy_avg[lang][step] = np.mean(entropy_list)

    # by train lang, average the entropies by each language by each step across models.
    for lang, step_dict in step2train_lang_entropy.items():
        for step, entropy_list in step_dict.items():
            step2train_lang_entropy_avg[lang][step] = np.mean(entropy_list)

    df_eval_lang_entropy = pd.DataFrame(step2eval_lang_entropy_avg).T
    df_train_lang_entropy = pd.DataFrame(step2train_lang_entropy_avg).T
    # print("train dataframe entropy")
    # print(df_train_lang_entropy)
    return df_eval_lang_entropy, df_train_lang_entropy


def get_corr_plot_for_eval(df, df_results_metric_lc_melted_avg, lang, lingual, level,
                           generation_setting,
                           by_entropy, inversion_metric, mode):
    # for eval
    df = df.sort_values("Base")
    df["lang"] = df.index

    # lang, step, entropy, language
    # if lang == "eval":
    df_melted = df.melt(id_vars='lang', var_name='step', value_name='entropy')
    df_melted["language"] = df_melted["lang"].map(iso2name)
    if lang == "train":
        df_results_metric_lc_melted_avg.rename(columns={"training_langs": "lang"}, inplace=True)
    # print("train", df_results_metric_lc_melted_avg)

    df_melted_inversion_metric = pd.merge(df_melted,
                                          df_results_metric_lc_melted_avg,
                                          on=["lang", "step"], how="left")

    # print("merged:", df_melted_inversion_metric)
    # get mt5 language percentages.
    df_melted_inversion_metric["mt5"] = df_melted_inversion_metric["lang"].map(
        lang2percent)

    df_melted_inversion_metric.to_csv(f"results/inversion_language_confusion/{level}_level/dataframes/{lingual}_{level}_{generation_setting}_{by_entropy}_{inversion_metric}_for_{lang}_{mode}.csv")
    df_corr = df_melted_inversion_metric[["entropy", "inversion_avg", "mt5"]]
    # print(df_corr.head(5))

    corr_annotated = get_annotated_corr(df_corr)
    # print(corr_annotated)
    corr_annotated.to_csv(
        f"results/inversion_language_confusion/{level}_level/corr/{lingual}_{level}_{generation_setting}_{by_entropy}_{inversion_metric}_for_{lang}_{mode}.csv")

    outputfile_plot = f"results/inversion_language_confusion/{level}_level/plots/{lingual}_{level}_{generation_setting}_{by_entropy}_{inversion_metric}_for_{lang}_{mode}.pdf"

    # print(f"plotting and saving to {outputfile_plot}")
    plot_lc_inversion(df_melted_inversion_metric, by_entropy, outputfile=outputfile_plot, lang=lang,
                      metric_name=inversion_metric,
                      generation_setting=generation_setting, level=level)


def measuring_language_confusion(lingual="multi", level="line", generation_setting="crosslingual",
                                 inversion_metric="bleu_score", by_entropy="weighted_entropy", mode=True):
    df_results_metric = get_inversion_results(metric=inversion_metric, lingual=lingual)

    # print(f"loading language confusion dataset {lingual} - {level} - {generation_setting}")
    df_lc = get_language_confusion_results(lingual=lingual, level=level, generation_setting=generation_setting, mode=mode)

    ## calculate entropies.
    # get entropy for all languages in pred langs on normalzied distributions
    if by_entropy == "entropy_all":
        df_lc[by_entropy] = df_lc["pred_langs_dist"].apply(get_entropy_for_all)

    # get entropy for all languages that are not train or eval languages
    elif by_entropy == "entropy_out":
        df_lc[by_entropy] = df_lc.apply(get_entropy_outside_eval_train, axis=1)

    # modified entropy
    elif by_entropy == "weighted_entropy":
        df_lc[by_entropy] = df_lc.apply(get_reweighted_entropy, axis=1)

    # print("entropy dataframe")
    df_lc.to_csv(
        f"results/inversion_language_confusion/{level}_level/dataframes/{lingual}_{level}_{generation_setting}_{by_entropy}_{mode}.csv")

    # print(df_lc)

    # get relevant df_result_metric re. df_lc dataframe
    df_results_metric_lc = df_results_metric[df_results_metric["model_name"].isin(df_lc["model"].tolist())]

    df_results_metric_lc = df_results_metric_lc.drop(columns=["model"])
    df_results_metric_lc_melted = pd.melt(df_results_metric_lc,
                                          id_vars=['model_name', 'step'],  # Columns to keep (model_name, step)
                                          var_name='lang',
                                          # New column for the language names (deu_Latn, mlt_Latn, etc.)
                                          value_name=f"{inversion_metric}")

    def replace_lang(x):
        return x.split("_")[0]

    df_results_metric_lc_melted["lang"] = df_results_metric_lc_melted["lang"].apply(replace_lang)

    ### aggregate for the training languages.
    model2training_languages = {'me5_jpn_Jpan': ["jpn"], 'me5_deu_Latn': ["deu"], 'me5_urd_pan': ["urd", "pan"],
                                'me5_urd_hin': ["urd", "hin"], 'me5_tur_guj': ["tur", "guj"],
                                'me5_urd_guj': ["urd", "guj"], 'me5_hin_guj': ["hin", "guj"],
                                'me5_tur_urd': ["tur", "urd"], 'me5_pan_Guru': ["pan"], 'me5_kaz_hin': ["kaz"],
                                'me5_hin_Deva': ["hin"], 'me5_heb_Hebr': ["heb"], 'me5_cmn_jpn': ["cmn", "jpn"],
                                'me5_mon_Cyrl': ["mon"], 'me5_latn-script': ["deu", "tur"],
                                'me5_tur_pan': ["tur", "pan"], 'me5_kaz_Cyrl': ["kaz"], 'me5_cmn_Hani': ["cmn"],
                                'me5_tur_Latn': ["tur"], 'me5_kaz_pan': ["kaz", "pan"],
                                'me5_urd_Arab': ["urd"], 'me5_tur_hin': ["tur"], 'me5_cyrl-script': ["kaz", "mon"],
                                'me5_hin_pan': ["hin", "pan"], 'me5_turkic-fami': ["tur", "kaz"],
                                'me5_guj_Gujr': ["guj"], 'me5_pan_guj': ["pan", "guj"], 'me5_kaz_guj': ["kaz", "guj"],
                                'me5_kaz_urd': ["kaz", "urd"], 'me5_arb_Arab': ["arb"],
                                'me5_heb_arb': ["heb", "arb"]}

    df_results_metric_lc_melted["training_langs"] = df_results_metric_lc_melted["model_name"].apply(
        lambda x: "_".join(sorted(model2training_languages[x])))
    df_results_metric_lc_melted_avg_for_train_lang = df_results_metric_lc_melted.groupby(['training_langs', 'step'],
                                                                                         as_index=False).agg(
        inversion_avg=(inversion_metric, 'mean'))

    print("dataframe metric ")

    print(df_results_metric_lc_melted_avg_for_train_lang)
    # average the inversion performance across the models for each langauge and each step.
    df_results_metric_lc_melted_avg = df_results_metric_lc_melted.groupby(['lang', 'step'], as_index=False).agg(
        inversion_avg=(inversion_metric, 'mean'))

    # print(df_results_metric_lc_melted_avg.head(2))

    # get entropy dataframe aggregated
    # lang, base, step1, step50+sbeam8
    # print(f"get entropy dataframe aggregated from language confusion")
    df_eval_lang_entropy, df_train_lang_entropy = get_step2eval_lang_entropy_dataframe(df_lc, by_entropy)
    # print(df_train_lang_entropy.head(2))
    # for eval


    get_corr_plot_for_eval(df_eval_lang_entropy, df_results_metric_lc_melted_avg, "eval", lingual, level,
                           generation_setting,
                           by_entropy, inversion_metric, mode)

    # this might not be correct.
    # TODO: CORRECT THE TRAIN LANGUAGE RESULTS.
    get_corr_plot_for_eval(df_train_lang_entropy, df_results_metric_lc_melted_avg_for_train_lang, "train", lingual,
                           level,
                           generation_setting,
                           by_entropy, inversion_metric, mode)


if __name__ == '__main__':
    # measuring_language_confusion(lingual="multi", level="line", generation_setting="crosslingual",
    #                              inversion_metric="bleu_score", by_entropy="weighted_entropy")
    inversion_metric = "bleu_score"
    # for lingual in ["multi"]:
    #     for level in ["line", "word"]:
    #         for generation_setting in ["crosslingual", "monolingual", "all"]:
    #             for by_entropy in ["weighted_entropy", "entropy_all", "entropy_out"]:
    #                 measuring_language_confusion(lingual=lingual, level=level, generation_setting=generation_setting,
    #                                              inversion_metric=inversion_metric, by_entropy=by_entropy, mode=True)
    for lingual in ["multi"]:
        for level in ["line", "word"]:
            for generation_setting in ["crosslingual"]:
                for by_entropy in ["weighted_entropy"]:
                    measuring_language_confusion(lingual=lingual, level=level, generation_setting=generation_setting,
                                                 inversion_metric=inversion_metric, by_entropy=by_entropy, mode=True)
