from time import sleep
from huggingface_hub import login as huggingface_login
import yaml

from datasets import load_dataset, DatasetDict


def login_huggingface():
    """login huggingface account for data."""
    access_token = "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"
    return huggingface_login(token=access_token)


login_huggingface()


def add_lang_id(example):
    example["text"] = f"[{example['lang'].split('_')[0]}] {example['text']}"
    return example


def add_lang_id_script_id(example):
    example["text"] = f"[{example['lang'].split('_')[0]}] [{example['lang'].split('_')[1]}] {example['text']}"
    return example


def processing_dataset_dict(dataset_name, lang_id, lang_script_id):
    # test dataset.
    dataset_dict = load_dataset(dataset_name)
    new_dataset_dict = dict()
    for lang in dataset_dict:
        dataset = dataset_dict[lang]
        if lang_id and not lang_script_id:
            new_dataset = dataset.map(add_lang_id)
        if lang_script_id:
            new_dataset = dataset.map(add_lang_id_script_id)
        new_dataset_dict[lang] = new_dataset
    dataset_new = DatasetDict(new_dataset_dict)
    return dataset_new


def processing_dataset(dataset_name, lang_id, lang_script_id):
    dataset = load_dataset(dataset_name)
    if lang_id and not lang_script_id:
        new_dataset = dataset.map(add_lang_id)
        print(new_dataset['train'][0])
        return new_dataset
    if lang_script_id:
        new_dataset = dataset.map(add_lang_id_script_id)
        print(new_dataset['train'][0])
        return new_dataset


def adding_lang_script_id():
    with open("lang2file.yaml") as f:
        lang2file = yaml.safe_load(f)

    #### mt_test_set
    dataset_name = "yiyic/mt_ms_test"
    print(f"processing dataset {dataset_name}....")

    new_dataset = processing_dataset_dict(dataset_name, True, False)
    new_dataset.push_to_hub(f"{dataset_name}_lang_id")
    sleep(3)
    new_dataset = processing_dataset_dict(dataset_name, True, True)
    new_dataset.push_to_hub(f"{dataset_name}_lang_script_id")
    #
    # for in-script datasets, only add lang id.
    for script in ["lat_scrp", "cyr_scrp", "ara_scrp"]:
        dataset_name = lang2file[script]
        print(f"processing dataset {dataset_name}....")
        print(f"processing train...")
        new_dataset = processing_dataset(dataset_name, True, False)
        new_dataset.push_to_hub(f"{dataset_name}_lang_id")
        sleep(3)

        # dev dataset
        dataset_dev = dataset_name.replace("_train", "_dev")
        new_dataset = processing_dataset(dataset_dev, True, False)
        new_dataset.push_to_hub(f"{dataset_dev}_lang_id")
        sleep(3)

    # for in-family datasets, add lang_script_id
    for lang_fam in ["ind_fami", "tur_fami", "atl_fami"]:
        dataset_name = lang2file[lang_fam]
        print(f"processing dataset {dataset_name}....")
        print(f"processing train...")
        new_dataset = processing_dataset(dataset_name, True, True)
        new_dataset.push_to_hub(f"{dataset_name}_lang_script_id")
        sleep(3)

        # dev dataset
        dataset_dev = dataset_name.replace("_train", "_dev")
        new_dataset = processing_dataset(dataset_dev, True, True)
        new_dataset.push_to_hub(f"{dataset_dev}_lang_script_id")
        sleep(3)


if __name__ == '__main__':
    adding_lang_script_id()
