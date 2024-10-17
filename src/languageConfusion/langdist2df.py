import pandas as pd
import json

evals = [
    'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
    'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
    'ydd_Hebr', 'heb_Hebr',
    'arb_Arab', 'urd_Arab',
    'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
    'cmn_Hani', 'jpn_Jpan', 'kor_Hang',
    'amh_Ethi']

lang2langscript = {x.split("_")[0]: x for x in evals}


def get_key2lang(model2langs):
    """
    Get the dictionary of model and their training data.
    """
    key2lang = {}
    for key in model2langs:
        # monolingual inversion models
        if key == "text2vec_cmn_Hani":
            key2lang[key] = ["cmn_Hani"]
        elif key == "gtr_deu_Latn":
            key2lang[key] = ["deu_Latn"]
        elif key == "alephbert_heb_Hebr":
            key2lang[key] = ["heb_Hebr"]

        # multilingual inversion models
        elif "me5_" in key:
            # me5_pan_Guru
            # me5_turkic-fami
            key_ = key.replace("me5_", "")

            if "_" in key_:
                if len(key_) == 7:
                    lang1, lang2 = key_.split("_")
                    key2lang[key] = [lang2langscript[lang1], lang2langscript[lang2]]

                if len(key_) == 8:
                    key2lang[key] = [key_]

            elif key_ == "turkic-fami":
                key2lang[key] = ['tur_Latn', 'kaz_Cyrl']

            elif key_ == "arab-script":
                key2lang[key] = ["arb_Arab", "heb_Hebr"]

            elif key_ == "latn-script":
                key2lang[key] = ["deu_Latn", "tur_Latn"]

            elif key_ == "cyrl-script":
                key2lang[key] = ["kaz_Cyrl", "mon_Cyrl"]

    return key2lang


def get_cos_similarity(filepath, mode):
    # get cosine similarities, to see if it can help predict languages
    with open(filepath) as f:
        data = json.load(f)
    model_names = []
    steps = []
    values = []
    eval_langs = []
    for model, eval_dict in data.items():
        if mode == "multi":
            model_name = model.replace("yiyic/mt5_", "").replace("_32_2layers", "").replace("_corrector", "").replace(
                "_inverter", "")
        elif mode == "mono":
            model_name = model.replace("yiyic/mt5_", "").replace("_32_corrector", "").replace(
                "_32_inverter", "")
        else:
            model_name = None

        for eval_lang, eval_steps in eval_dict.items():
            for step, value in eval_steps.items():
                if step == "Step50_sbeam8":
                    steps.append("Step50+sbeam8")
                else:
                    steps.append(step)
                values.append(value)
                model_names.append(model_name)
                eval_langs.append(eval_lang)
    df = pd.DataFrame.from_dict({
        "model": model_names,
        "eval_lang": eval_langs,
        "step": steps,
        "emb_cos_sim": values
    })
    print(df.head())
    return df


def get_model2langs(langdist_file, mode):
    with open(langdist_file) as f:
        langdist = json.load(f)

    outputfile = langdist_file.replace(".json", ".csv")

    model2langs = dict()
    for model, langs_dict in langdist.items():
        if "mt5_me5_" in model:
            # me5_pan_Guru
            model_name = model.replace("mt5_", "").replace("_32_2layers_corrector", "").replace(
                "_32_2layers_inverter", "")
            # print(model_name)

            if model_name not in model2langs:
                if model_name == "ara-script":
                    model2langs["arab-script"] = dict()
                    model_name = "arab-script"
                else:
                    model2langs[model_name] = dict()

            for eval_lang, steps_eval in langs_dict.items():
                if eval_lang not in model2langs[model_name]:
                    model2langs[model_name][eval_lang] = dict()

                for step, eval in steps_eval.items():
                    if step not in model2langs[model_name][eval_lang]:
                        model2langs[model_name][eval_lang][step] = eval
        else:
            # mt5_alephbert_heb_Hebr_32_corrector
            model_name = model.replace("mt5_", "").replace("_32_corrector", "").replace(
                "_32_inverter", "")
            print(model_name)

            if model_name not in model2langs:
                model2langs[model_name] = dict()

            for eval_lang, steps_eval in langs_dict.items():
                if eval_lang not in model2langs[model_name]:
                    model2langs[model_name][eval_lang] = dict()

                for step, eval in steps_eval.items():
                    if step not in model2langs[model_name][eval_lang]:
                        model2langs[model_name][eval_lang][step] = eval

    print(len(model2langs))
    print(model2langs.keys())
    # output to model2langs.json
    # get the training languages for model.
    key2lang = get_key2lang(model2langs)
    print(len(key2lang))
    print(set(model2langs.keys()).difference(key2lang.keys()))

    model2langs_dict = dict()

    for key, langs in key2lang.items():
        if key not in model2langs_dict:
            model2langs_dict[key] = dict()
        model2langs_dict[key]["training"] = langs
        model2langs_dict[key]["langdict"] = model2langs[key]

    # print(model2langs_dict)

    training_data_list = []  # [cmn_Hani, jpn_Jpan]
    # training_data_labels = []  # {"deu_Latn":0.68}...
    eval_langs = []  # deu_Latn
    steps = []  # base, step1, step50+sbeam8
    pred_langs = []  # {'deu_Latn': 1.0}
    inversion_models = []

    for model in model2langs_dict:
        training_data = model2langs_dict[model]["training"]
        langdict = model2langs_dict[model]["langdict"]

        for eval_lang, step_eval in langdict.items():
            # deu_latn, xxxx
            for step, eval in step_eval.items():
                # labels

                inversion_models.append(model)
                training_data_list.append(training_data)
                eval_langs.append(eval_lang)
                steps.append(step)
                pred_langs.append(eval)

    df_lang = pd.DataFrame({"model": inversion_models, "training": training_data_list,
                            "eval_lang": eval_langs, "step": steps,
                            "pred_langs": pred_langs})
    df_lang.to_csv(outputfile, index=False)

    if mode == "mono":
        df_cos = get_cos_similarity("results/mt5_mono/monolingual_eval_emb_cos_sim.json", mode)
    elif mode == "multi":
        df_cos = get_cos_similarity("results/mt5_me5/multilingual_eval_emb_cos_sim.json", mode)
    elif mode == "mono+multi":
        df_cos_mono = get_cos_similarity("results/mt5_mono/monolingual_eval_emb_cos_sim.json", "mono")
        df_cos_multi = get_cos_similarity("results/mt5_me5/multilingual_eval_emb_cos_sim.json", "multi")
        df_cos = pd.concat([df_cos_mono, df_cos_multi])

    print(df_cos.head())
    df_lang_ = pd.merge(df_lang, df_cos, on=["model", "step", "eval_lang"], how="left")
    print(len(df_lang_))

    df_lang_.to_csv(outputfile, index=False)

    return df_lang


if __name__ == '__main__':
    import plac

    for mode in ["multi", "mono", "mono+multi"]:
        print(mode)
        print("*" * 20)
        filepath = f"language_confusion/langdist_data/dataset2langdist_line_level_{mode}.json"
        get_model2langs(filepath, mode)
        print("*" * 20)
        filepath_word = f"language_confusion/langdist_data/dataset2langdist_word_level_{mode}.json"
        get_model2langs(filepath_word, mode)
