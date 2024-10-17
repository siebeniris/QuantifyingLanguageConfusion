import json
import os

import numpy as np
import pandas as pd
from collections import defaultdict
from evaluations.model_lists import model_list_corrector, model_list_inverter, eval_langs, model_list_inverter_mono, \
    model_list_corrector_mono


def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def collect_through_eval_logs(lingual="multilingual", metric="token_set_f1"):
    folder = f"eval_logs/{lingual}"
    set_tokens_f1_dict = defaultdict(dict)
    for file in os.listdir(folder):
        # go through json files "eval_logs/multilingual/eval_mt5_me5_ara-script_32_2layers_corrector.json"
        filepath = os.path.join(folder, file)
        if file.endswith(".json"):

            eval_log_model = load_json_file(filepath)
            if eval_log_model["model"] is not None:
                model_output = eval_log_model["model"].replace("yiyic/", "yiyic__")
                model_name = model_output.replace("yiyic__", "")
                # model_name = model_output.replace("yiyic__mt5_", "").replace("_32_2layers", "").replace("_32","")

                if model_name not in set_tokens_f1_dict:
                    set_tokens_f1_dict[model_name] = dict()

                if os.path.exists(os.path.join("saves", model_output)):
                    if "inverter" in model_output:
                        print(f"processing {model_name}")
                        for eval in eval_log_model["evaluations"]:
                            dataset = eval["dataset"]
                            if dataset not in set_tokens_f1_dict[model_name]:
                                set_tokens_f1_dict[model_name][dataset] = dict()

                            decoded_folder = eval["embeddings_file"]
                            set_eval_file = os.path.join(decoded_folder, "set_token_eval.json")
                            if "Base" not in set_tokens_f1_dict[model_name][dataset]:
                                if os.path.exists(set_eval_file):
                                    eval_result = load_json_file(set_eval_file)[metric]
                                    if metric in ["token_set_f1"]:
                                        set_tokens_f1_dict[model_name][dataset]["Base"] = round(eval_result * 100, 2)
                                    else:
                                        set_tokens_f1_dict[model_name][dataset]["Base"] = eval_result
                    elif "corrector" in model_output:
                        print(f"processing {model_name}")
                        if "semitic-fami_" not in model_name:
                            for eval, eval_results in eval_log_model["evaluations"].items():
                                # deu, results
                                for eval_step, eval_steps_results in eval_results.items():
                                    # step1, files_output
                                    if eval not in set_tokens_f1_dict[model_name]:
                                        set_tokens_f1_dict[model_name][eval] = dict()

                                    if eval_step == "steps 1":
                                        if "Step1" not in set_tokens_f1_dict[model_name][eval]:
                                            if "embeddings_files" in eval_steps_results:
                                                decoded_folder_step1 = eval_steps_results["embeddings_files"]
                                                set_eval_file_step1 = os.path.join(decoded_folder_step1,
                                                                                   "set_token_eval.json")
                                                if os.path.exists(set_eval_file_step1):
                                                    eval_result_step1 = load_json_file(set_eval_file_step1)[metric]
                                                    if metric in ["token_set_f1"]:
                                                        set_tokens_f1_dict[model_name][eval]["Step1"] = round(
                                                            eval_result_step1 * 100, 2)
                                                    else:
                                                        set_tokens_f1_dict[model_name][eval]["Step1"] = eval_result_step1

                                    if eval_step.endswith("beam width 8"):
                                        if "Step50_sbeam8" not in set_tokens_f1_dict[model_name][eval]:
                                            if "embeddings_files" in eval_steps_results:
                                                decoded_folder_b8 = eval_steps_results["embeddings_files"]
                                                set_eval_file_b8 = os.path.join(decoded_folder_b8, "set_token_eval.json")

                                                if os.path.exists(set_eval_file_b8):
                                                    set_eval_b8_result = load_json_file(set_eval_file_b8)[metric]
                                                    if metric in ["token_set_f1"]:
                                                        set_tokens_f1_dict[model_name][eval]["Step50_sbeam8"] = round(
                                                            set_eval_b8_result * 100, 2)
                                                    else:
                                                        set_tokens_f1_dict[model_name][eval][
                                                            "Step50_sbeam8"] = set_eval_b8_result

    # with open(os.path.join(outputfolder, f'{lingual}_eval_{metric}.json'), "w") as f:
    #     json.dump(set_tokens_f1_dict, f)
    return set_tokens_f1_dict


def dict2df(lingual="multilingual", metric="token_set_f1", outputfolder="results/mt5_me5"):
    if lingual == "multilingual":

        set_tokens_f1_dict = collect_through_eval_logs(lingual, metric)

        base_dict = defaultdict(dict)
        step1_dict = defaultdict(dict)
        step50_b8_dict = defaultdict(dict)

        for model, eval_results in set_tokens_f1_dict.items():
            if "inverter" in model:
                for eval_dataset, result in eval_results.items():
                    base_dict[model][eval_dataset] = result.get("Base", np.nan)
            if "corrector" in model:
                for eval_dataset, result in eval_results.items():
                    step1_dict[model][eval_dataset] = result.get("Step1", np.nan)
                    step50_b8_dict[model][eval_dataset] = result.get("Step50_sbeam8", np.nan)

        base_df = pd.DataFrame.from_records(base_dict).T
        base_df = base_df.reindex(columns=eval_langs)
        base_df = base_df.reindex(index=model_list_inverter)
        base_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_base_inverter.csv"))

        step1_df = pd.DataFrame.from_records(step1_dict).T
        step1_df = step1_df.reindex(columns=eval_langs)
        step1_df = step1_df.reindex(index=model_list_corrector)
        step1_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_step1_corrector.csv"))

        step50_df = pd.DataFrame.from_dict(step50_b8_dict).T
        step50_df = step50_df.reindex(columns=eval_langs)
        step50_df = step50_df.reindex(index=model_list_corrector)
        step50_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_step50_sbeam8_corrector.csv"))
    else:
        set_tokens_f1_dict = collect_through_eval_logs(lingual, metric)
        evals = ['deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
                 'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
                 'ydd_Hebr', 'heb_Hebr',
                 'arb_Arab', 'urd_Arab',
                 'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
                 'cmn_Hani', 'jpn_Jpan', 'kor_Hang',
                 'amh_Ethi']

        base_dict = defaultdict(dict)
        step1_dict = defaultdict(dict)
        step50_b8_dict = defaultdict(dict)

        for model, eval_results in set_tokens_f1_dict.items():
            if "inverter" in model:
                for eval_dataset, result in eval_results.items():
                    base_dict[model][eval_dataset] = result.get("Base", np.nan)
            if "corrector" in model:
                for eval_dataset, result in eval_results.items():
                    step1_dict[model][eval_dataset] = result.get("Step1", np.nan)
                    step50_b8_dict[model][eval_dataset] = result.get("Step50_sbeam8", np.nan)
        base_df = pd.DataFrame.from_records(base_dict).T
        print(base_df)
        base_df = base_df.reindex(columns=evals)
        base_df = base_df.reindex(index=model_list_inverter_mono)
        base_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_base_inverter.csv"))

        step1_df = pd.DataFrame.from_records(step1_dict).T
        step1_df = step1_df.reindex(columns=evals)
        step1_df = step1_df.reindex(index=model_list_corrector_mono)
        step1_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_step1_corrector.csv"))

        step50_df = pd.DataFrame.from_dict(step50_b8_dict).T
        step50_df = step50_df.reindex(columns=evals)
        step50_df = step50_df.reindex(index=model_list_corrector_mono)
        step50_df.to_csv(os.path.join(outputfolder, f"{lingual}_eval_{metric}_step50_sbeam8_corrector.csv"))


if __name__ == '__main__':
    import plac

    plac.call(dict2df)
