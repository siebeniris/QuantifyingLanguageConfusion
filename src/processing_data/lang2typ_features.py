import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances, cosine_similarity
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder


def crop_wals(gb_df, perc):
    """Remove languages from dataframe that do not have at least <perc>% feature coverage"""
    rows_list=[]
    tot_feats = len([x for x in gb_df.columns])
    for i, row in gb_df.iterrows():
        no_data = row.to_list().count(-1)
        # print(i, no_data)
        if (tot_feats - no_data) >=(perc * tot_feats):
            rows_list.append(row)
            # try:
            #     gb_df = gb_df.drop(i)
            # except Exception as msg:
            #     print(msg)

    return pd.concat(rows_list, axis=1)


df_wals_features_cropped = crop_wals(df_wals_features, 0.25)