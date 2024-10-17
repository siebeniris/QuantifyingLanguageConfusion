import os

import pandas as pd
import json

from ast import literal_eval
from itertools import product
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

eval_langs = [
    'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
    'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
    'ydd_Hebr', 'heb_Hebr',
    'arb_Arab', 'urd_Arab',
    'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
    'cmn_Hani', 'jpn_Jpan', 'kor_Hang', 'amh_Ethi'
]

lang2fam = {
    'amh_Ethi': 'semitic',
    'arb_Arab': 'semitic',
    'cmn_Hani': 'atlatic',
    'deu_Latn': 'germanic',
    'fin_Latn': 'uralic',
    'guj_Gujr': 'indo',
    'heb_Hebr': 'semitic',
    'hin_Deva': 'indo',
    'hun_Latn': 'uralic',
    'jpn_Jpan': 'atlatic',
    'kaz_Cyrl': 'turkic',
    'kor_Hang': 'atlatic',
    'mhr_Cyrl': 'uralic',
    'mlt_Latn': 'semitic',
    'mon_Cyrl': 'atlatic',
    'pan_Guru': 'indo',
    'sin_Sinh': 'indo',
    'tur_Latn': 'turkic',
    'urd_Arab': 'indo',
    'ydd_Hebr': 'germanic'

}

word_order_langs = {
    'amh_Ethi': 'SOV',  # WALS
    'cmn_Hani': 'SVO',  # WALS
    'deu_Latn': 'ND',   # WALS - non-dominant
    'fin_Latn': 'SVO',  # wals
    'guj_Gujr': 'SOV',  # wals
    'hin_Deva': 'SOV',  # wals
    'hun_Latn': 'ND',  # wals
    'jpn_Jpan': 'SOV',  # wals
    'kaz_Cyrl': 'SOV',  # wikipedia, https://en.wikipedia.org/wiki/Kazakh_language
    'kor_Hang': 'SOV',  # wals
    'mhr_Cyrl': 'SOV',  # mari # https://meadow-mari.web-corpora.net/index_en.html#:~:text=The%20default%20word%20order%20in,subject%20–%20object%20–%20verb).
    'mlt_Latn': 'ND',  # non-dominant.
    'mon_Cyrl': 'SOV',  # WIKIPEDIA
    'pan_Guru': 'SOV',  # WALS
    'sin_Sinh': 'SOV',  # WALS
    'tur_Latn': 'SOV',  # wals
    'urd_Arab': 'SOV',  # wals
    'heb_Hebr': 'SVO',  # mordern Hebrew
    'arb_Arab': 'VSO',  # modern Arabic
    'ydd_Hebr': 'SVO'
}


script_writting={
    'amh_Ethi': 'LTR',
    'cmn_Hani': 'LTR',
    'deu_Latn': 'LTR',
    'fin_Latn': 'LTR',
    'guj_Gujr': 'LTR',
    'hin_Deva': 'LTR',
    'hun_Latn': 'LTR',
    'jpn_Jpan': 'LTR',
    'kaz_Cyrl': 'LTR',
    'kor_Hang': 'LTR',
    'mhr_Cyrl': 'LTR',  # mari
    'mlt_Latn': 'LTR',
    'mon_Cyrl': 'LTR',
    'pan_Guru': 'LTR',
    'sin_Sinh': 'LTR',
    'tur_Latn': 'LTR',

    'urd_Arab': 'RTL',
    'heb_Hebr': 'RTL',
    'arb_Arab': 'RTL',
    'ydd_Hebr': 'RTL'
}


def get_lang2lang_word_order_dict(eval_langs_list):
    lang2lang_word_order = dict()
    for lang1, lang2 in product(eval_langs_list, repeat=2):

        if lang1 not in lang2lang_word_order:
            lang2lang_word_order[lang1] = dict()

        if word_order_langs[lang1] == "ND" or word_order_langs[lang2] == "ND":
            lang2lang_word_order[lang1][lang2] = 1
        elif word_order_langs[lang1] == word_order_langs[lang2]:
            lang2lang_word_order[lang1][lang2] = 1
        else:
            lang2lang_word_order[lang1][lang2] = 0
    return lang2lang_word_order


def get_lang2lang_family_dict(eval_langs_list):
    lang2lang_family = dict()
    for lang1, lang2 in product(eval_langs_list, repeat=2):

        if lang1 not in lang2lang_family:
            lang2lang_family[lang1] = dict()

        if lang2fam[lang1] == lang2fam[lang2]:
            lang2lang_family[lang1][lang2] = 1
        else:
            lang2lang_family[lang1][lang2] = 0
    return lang2lang_family


def get_lang2lang_script_dict(eval_langs_list):
    """
    get lang2lang script dict from pairs of languages.
    For example, arb_Arab and urd_Arab has the same script, then it outputs 1.
    """
    lang2lang_script = dict()
    for lang1, lang2 in product(eval_langs_list, repeat=2):
        lang1_script = lang1.split("_")[1]
        lang2_script = lang2.split("_")[1]
        if lang1_script == lang2_script:
            if lang1 not in lang2lang_script:
                lang2lang_script[lang1] = dict()
            lang2lang_script[lang1][lang2] = 1
        else:
            if lang1 not in lang2lang_script:
                lang2lang_script[lang1] = dict()
            lang2lang_script[lang1][lang2] = 0
    return lang2lang_script


def get_lang2lang_lr_dict(eval_langs_list):
    """
    Get whether two languages are written in the same order of script, ltr or rtl
    """
    lang2lang_lr = dict()
    for lang1, lang2 in product(eval_langs_list, repeat=2):

        if lang1 not in lang2lang_lr:
            lang2lang_lr[lang1] = dict()

        if script_writting[lang1] == script_writting[lang2]:
            lang2lang_lr[lang1][lang2] = 1
        else:
            lang2lang_lr[lang1][lang2] = 0
    return lang2lang_lr


def preprocessing_data_for_modeling(file, level="line", mode="mono"):
    output_dir = f"language_confusion/data/{level}/{mode}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # reading the model2langs.csv
    df_lang = pd.read_csv(file)
    df_lang['pred_langs'] = df_lang['pred_langs'].apply(literal_eval)
    # drop the empty cells
    df_lang = df_lang[df_lang["pred_langs"] != {}]
    # cos_sim
    # df_lang = df_lang.dropna(subset=["emb_cos_sim"])

    # get all the languages from
    all_langs = set(lang for preds in df_lang['pred_langs'] for lang in preds.keys())
    print(f"{len(all_langs)} languages in the dataset")

    # make languages numerical
    label_encoder_eval = LabelEncoder()
    df_lang["eval_lang_encoded"] = label_encoder_eval.fit_transform(df_lang["eval_lang"])

    # training languages. a list.
    df_lang["training"] = df_lang["training"].apply(literal_eval)
    mlb = MultiLabelBinarizer()
    training_encoded = mlb.fit_transform(df_lang['training'])
    training_encoded_df = pd.DataFrame(training_encoded, columns=mlb.classes_)

    # reset index of the dataframe.
    df_lang = df_lang.reset_index().drop(columns=["index"])
    print(f"df_lang length {len(df_lang)}")

    eval_langs_list = eval_langs

    ################################
    # ablation1. script
    lang2lang_script_dict = get_lang2lang_script_dict(eval_langs_list)

    def get_script_feature(x, y):
        same_script = 0
        for lang in x:
            if lang2lang_script_dict[lang][y] == 1:
                same_script += 1
        if same_script > 0:
            return 1
        else:
            return 0

    df_lang["script"] = df_lang.apply(lambda x: get_script_feature(x["training"], x["eval_lang"]), axis=1)

    ###################################
    # ablation2. family
    lang2lang_family_dict = get_lang2lang_family_dict(eval_langs_list)

    def get_family_feature(x, y):
        same_family = 0
        for lang in x:
            if lang2lang_family_dict[lang][y] == 1:
                same_family += 1
        if same_family > 0:
            return 1
        else:
            return 0

    df_lang["family"] = df_lang.apply(lambda x: get_family_feature(x["training"], x["eval_lang"]), axis=1)

    ############################################
    # ablation3. word order

    lang2lang_word_order_dict = get_lang2lang_word_order_dict(eval_langs_list)

    def get_word_order_feature(x, y):
        same_word_order =0
        for lang in x:
            if lang2lang_word_order_dict[lang][y] == 1:
                same_word_order += 1
        if same_word_order > 0:
            return 1
        else:
            return 0

    df_lang["word_order"] = df_lang.apply(lambda x: get_word_order_feature(x["training"], x["eval_lang"]), axis=1)

    ############################################
    # ablation3. script ltr
    # training data themselves.
    lang2lang_lr = get_lang2lang_lr_dict(eval_langs_list)

    def get_script_lr(x, y):
        same_lr = 0
        for lang in x:
            if lang2lang_lr[lang][y] == 1:
                same_lr += 1
        if same_lr > 0:
            return 1
        else:
            return 0

    df_lang["script_lr"] = df_lang.apply(lambda x: get_script_lr(x["training"], x["eval_lang"]), axis=1)

    def get_script_lr_for_training_data(x):
        if len(x) == 1:
            return 1
        else:
            lang1, lang2 = x[0], x[1]
            if lang2lang_lr[lang1][lang2] == 1:
                return 1
            else:
                return 0

    df_lang["training_script_lr"] = df_lang["training"].apply(get_script_lr_for_training_data)

    ###########################################################################
    # df_lang = df_lang[df_lang["crosslingual"]]

    # get training dataframe.
    df_train = pd.concat([df_lang, training_encoded_df], axis=1)

    # get one hot vectors for the training steps.
    onehot_encoder = OneHotEncoder(sparse=False)
    step_encoded = onehot_encoder.fit_transform(df_lang[['step']])
    step_encoded_df = pd.DataFrame(step_encoded, columns=onehot_encoder.get_feature_names_out(['step']))

    df_train = pd.concat([df_train, step_encoded_df], axis=1)
    df_train = df_train.dropna()
    print(df_train)
    print(len(df_train))

    # probability of languages.
    probs_df = pd.DataFrame(df_train['pred_langs'].tolist(), index=df_train.index).fillna(0)
    # other languages probabilities
    probs_df["other"] = 1 - probs_df.sum(axis=1)

    probs_df.to_csv(f"{output_dir}/y_data.csv", index=False)
    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)

    X = df_train.drop(columns=['model', 'eval_lang', 'step', 'pred_langs', "training"])

    y = probs_df
    languages = list(y.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # save train/test data
    X_train.to_csv(f"{output_dir}/X_train.csv")
    y_train.to_csv(f"{output_dir}/y_train.csv")
    X_test.to_csv(f"{output_dir}/X_test.csv")
    y_test.to_csv(f"{output_dir}/y_test.csv")

    with open(f"{output_dir}/languages.json", "w") as f:
        json.dump(languages, f)


if __name__ == '__main__':
    for mode in ["multi", "mono", "mono+multi"]:
        print(mode)
        print("*" * 20)
        filepath = f"language_confusion/langdist_data/dataset2langdist_line_level_{mode}.csv"
        preprocessing_data_for_modeling(filepath, "line_level", mode)
        print("*" * 20)
        filepath_word = f"language_confusion/langdist_data/dataset2langdist_word_level_{mode}.csv"
        preprocessing_data_for_modeling(filepath_word, "word_level", mode)

