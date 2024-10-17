import os
import json
from ast import literal_eval
from itertools import combinations, chain
from tqdm import tqdm
import numpy as np

from itertools import product
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

glotto2code = {'amha1245': 'amh',
               'east2295': 'ydd',
               'east2328': 'mhr',
               'finn1318': 'fin',
               'guja1252': 'guj',
               'hebr1245': 'heb',
               'hind1269': 'hin',
               'hung1274': 'hun',
               'kaza1248': 'kaz',
               'kore1280': 'kor',
               'malt1254': 'mlt',
               'mand1415': 'cmn',
               'mong1331': 'mon',
               'nucl1301': 'tur',
               'nucl1643': 'jpn',
               'panj1256': 'pan',
               'sinh1246': 'sin',
               'stan1295': 'deu',
               'stan1318': 'arb',
               'urdu1245': 'urd'}

iso3_codes = list(glotto2code.values())


def get_feature_dict_colex(df_typ, typology):
    """
    Get the typological feature dictionary from dataframe.
    """
    if typology == "clics3":
        df_typ = df_typ[df_typ["Target"].isin(glotto2code)]
        df_typ = df_typ[df_typ["Source"].isin(glotto2code)]
    elif typology == "wn":
        df_typ = df_typ[df_typ["Target"].isin(iso3_codes)]
        df_typ = df_typ[df_typ["Source"].isin(iso3_codes)]

    langs = list(set(df_typ.Target.tolist()).intersection(set(df_typ.Source.tolist())))
    print(f"{len(langs)} Languages: {len(df_typ)} {typology} Features ")
    feature_dict = {}
    if typology == "clics3":

        df_typ["Target_iso3"] = df_typ["Target"].map(glotto2code)
        df_typ["Source_iso3"] = df_typ["Source"].map(glotto2code)

        for t, s, feature_value in zip(df_typ["Target_iso3"], df_typ["Source_iso3"], df_typ["pmi_norm"]):
            feature_dict[tuple(sorted([t, s]))] = feature_value

    else:
        for t, s, feature_value in zip(df_typ["Target"], df_typ["Source"], df_typ["pmi_norm"]):
            feature_dict[tuple(sorted([t, s]))] = feature_value

    return feature_dict


def compare_feature_each(i, j):
    """
    Compare feature element wise.
    """
    if i != -1 and j != -1:
        if i == j:
            return 1
        else:
            return 0
    else:
        return -1


def get_feature_dict_typology(df_typ, features):
    # the index is a list of langauges.
    langs = df_typ.index.tolist()
    df_typ = df_typ[features]
    # get the lang and their feature vectors.
    lang2feat_vecs = {}
    for lang, row in df_typ.iterrows():
        row_features = row.to_numpy()
        lang2feat_vecs[lang] = row_features

    # feature dictionary with tuple (lang1,lang2) and their vectors.
    feature_dict = {}
    for t in tqdm(combinations(langs, 2)):
        p = sorted(t)
        l1, l2 = p
        feature_dict[tuple(p)] = list(map(compare_feature_each, lang2feat_vecs[l1], lang2feat_vecs[l2]))
    return feature_dict


def processing_one_inversion_result_file(filepath, typology):
    # filepath: datasets/inversion_language_confusion/langdist_data/
    assert typology in ["clics3", "wn", "wals", "grambank"]
    df = pd.read_csv(filepath)
    outputfolder = os.path.dirname(filepath)
    # drop these two, the feature can be duplicated.
    df = df.drop(columns=["word_order", "emb_cos_sim"])
    df["training"] = df["training"].apply(literal_eval)
    df["training_iso3"] = df["training"].apply(lambda i: [x.split("_")[0] for x in i])
    df["eval_lang_iso3"] = df["eval_lang"].apply(lambda x: x.split("_")[0])

    if typology == "grambank":
        df_typ = pd.read_csv("datasets/inversion_language_confusion/gb_features.csv")
        df_typ.index = df_typ["iso639P3code"]
        features = [x for x in df_typ.columns if x.startswith("GB")]
        print(f"{len(df_typ)} Languages: {len(features)} {typology} Features ")
        feature_dict = get_feature_dict_typology(df_typ, features)

    elif typology == "wals":
        df_typ = pd.read_csv("datasets/inversion_language_confusion/wals_features.csv")
        df_typ_temp = df_typ.drop(columns=["iso639P3code", "Family", "Glottocode"])
        df_typ.index = df_typ["iso639P3code"]
        features = [x for x in df_typ_temp.columns]
        print(f"{len(df_typ)} Languages: {len(features)} {typology} Features ")
        feature_dict = get_feature_dict_typology(df_typ, features)

    elif typology == "clics3":
        df_typ = pd.read_csv("datasets/data_for_graph/clics3.csv")
        features = ["clics3_pmi_norm"]
        feature_dict = get_feature_dict_colex(df_typ, typology)

    elif typology == "wn":
        df_typ = pd.read_csv("datasets/data_for_graph/wn_colex.csv")
        features = ["wn_pmi_norm"]
        feature_dict = get_feature_dict_colex(df_typ, typology)

    ################################################################
    ################################################################
    def compare_elementwise(i, j):
        """
        compare for inversion results
        """
        if i != -1 and j != -1:
            if i == 1 or j == 1:
                return 1
            else:
                return 0
        else:
            return -1

    def get_feature_value(x, y):
        # compare features between training language and eval language.
        feature_list = []
        for i in x:
            p = tuple(sorted([i, y]))
            if p in feature_dict:
                feature_list.append(feature_dict[p])

        if len(feature_list) == 2:
            return list(map(compare_elementwise, feature_list[0], feature_list[1]))

        elif len(feature_list) == 1:
            return feature_list[0]
        else:
            return [-1 for _ in range(len(features))]

    def get_feature_value_colex(x, y):
        feature_list = []
        for i in x:
            p = tuple(sorted([i, y]))
            if p in feature_dict:
                feature_list.append(feature_dict[p])
        if len(feature_list) == 1:
            return feature_list
        elif len(feature_list) == 0:
            return [-1]
        else:
            # print(feature_list)
            return [np.mean(feature_list)]

    ################################################################
    ################################################################

    if typology in ["wals", "grambank"]:
        df[features] = pd.DataFrame(
            df.apply(lambda row: get_feature_value(row["training_iso3"], row["eval_lang_iso3"]), axis=1).tolist())
        df["check_all_minus_one"] = (df[features] == -1).all(axis=1)
        df = df[df["check_all_minus_one"] == False]
        print(df)
    elif typology in ["clics3", "wn"]:
        feature_col = f"{typology}_pmi_norm"
        print(features)
        df[features] = pd.DataFrame(
            df.apply(lambda row: get_feature_value_colex(row["training_iso3"], row["eval_lang_iso3"]), axis=1).tolist())

    df.to_csv(os.path.join(outputfolder, f"train_data_{typology}.csv"), index=False)


if __name__ == '__main__':
    import plac

    plac.call(processing_one_inversion_result_file)
