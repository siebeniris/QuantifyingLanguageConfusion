import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import LabelEncoder

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def do_regression(filepath, typology, outputfile):
    df = pd.read_csv(filepath)
    if typology == "clics3":
        df["clics3_pmi_norm"] = df["clics3_pmi_norm"].replace({-1: np.nan})
        # dropna on the pmi norm
        df = df.dropna(subset="clics3_pmi_norm")
        print(f"{len(df)} samples , {len(df.columns)} features")
    elif typology == "wn":
        df["wn_pmi_norm"] = df["wn_pmi_norm"].replace({-1: np.nan})
        df = df.dropna(subset="wn_pmi_norm")
        print(f"{len(df)} samples , {len(df.columns)} features")
    else:
        print(f"{len(df)} samples , {len(df.columns)} features")

    # convert to dict.
    df['pred_langs'] = df['pred_langs'].apply(literal_eval)

    label_encoder_eval = LabelEncoder()
    # delete the eval step for labels.
    df = df[df["step"] != "Labels"]
    df["eval_step"] = label_encoder_eval.fit_transform(df["step"])

    # X and Y
    if typology in ["clics3", "wn"]:
        df = df.drop(columns=['model', 'eval_lang', 'step', "training",
                              "step_Base", "step_Labels", "step_Step1", "step_Step50+sbeam8",
                              "training_iso3", 'eval_lang_iso3'])
    elif typology in ["grambank", "wals"]:
        df = df.drop(columns=['model', 'eval_lang', 'step', "training",
                              "step_Base", "step_Labels", "step_Step1", "step_Step50+sbeam8",
                              "training_iso3", 'eval_lang_iso3',
                              ])

    df = df.loc[:, (df != 0).any(axis=0)]
    print(f"after cleaning {len(df)} samples , {len(df.columns)} features")


    X = df.drop(columns=['pred_langs'])

    # get the targets
    probs_df = pd.DataFrame(df['pred_langs'].tolist(), index=df.index).fillna(0)
    probs_df["other"] = 1 - probs_df.sum(axis=1)

    y = probs_df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"running the Random Forest")
    regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    features = X_train.columns
    targets = y_train.columns

    # Create a dictionary to hold the mapping of target to feature importances
    target_feature_importances = {}

    # Accessing feature importances for each target
    for i, estimator in enumerate(regr.estimators_):
        target_name = targets[i]  # Assign a target name (e.g., target1, target2)
        # Map feature names to their importances for this target
        feature_importance_map = dict(zip(features, estimator.feature_importances_))
        # Add to the dictionary
        target_feature_importances[target_name] = feature_importance_map

    df_results = pd.DataFrame(target_feature_importances)
    df_results.to_csv(outputfile)

    languages = y.columns
    langs = []
    mse_values_list = []
    mae_values_list = []
    for lang, mse_values, mae_values in zip(languages, mse, mae):
        # print(f"MSE for {lang}: {mse_values:.5f}")
        langs.append(lang)
        mse_values_list.append(mse_values)
        mae_values_list.append(mae_values)

    langs.append("avg")
    mse_values_list.append(np.mean(mse))
    mae_values_list.append(np.mean(mae))

    df_mse = pd.DataFrame.from_dict({
        "lang": langs,
        "mse": mse_values_list,
        "mae": mae_values_list

    })
    print("avg:", np.mean(mse), " | ", np.mean(mae))


if __name__ == '__main__':
    import plac

    plac.call(do_regression)
