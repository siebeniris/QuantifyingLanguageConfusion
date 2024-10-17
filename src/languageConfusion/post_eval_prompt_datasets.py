import functools
import os
import string
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
from ftlangdetect import detect as ftdetect
import collections
from collections import Counter
import jieba
import nltk
import json
from tqdm import tqdm

import hebrew_tokenizer
from fugashi import Tagger

japaneseTagger = Tagger("-Owakati")
tk = nltk.WordPunctTokenizer()
from typing import List, Union
import scipy

from kiwipiepy import Kiwi

kiwi = Kiwi()

# load language codes from iso2 to iso3.
with open("src/languageConfusion/letter_codes.json") as f:
    lang_codes = json.load(f)

########################Language Detector ########################################
# NO YIDDISH, AMHARIC, SIN
# NO MALTESE (TRAIN)
languages = [Language.ENGLISH, Language.GERMAN, Language.HEBREW, Language.ARABIC,
             Language.HINDI, Language.GUJARATI, Language.PUNJABI, Language.URDU,
             Language.TURKISH, Language.KAZAKH,
             Language.CHINESE, Language.MONGOLIAN, Language.KOREAN, Language.JAPANESE,
             Language.HUNGARIAN, Language.FINNISH]

lang2idx = {
    Language.ENGLISH: "eng_Latn", Language.GERMAN: "deu_Latn", Language.HEBREW: "heb_Hebr",
    Language.ARABIC: "arb_Arab", Language.HINDI: "hin_Deva", Language.GUJARATI: "guj_Gujr",
    Language.PUNJABI: "pan_Guru", Language.URDU: "urd_Arab",
    Language.TURKISH: "tur_Latn", Language.KAZAKH: "kaz_Cyrl",
    Language.CHINESE: "cmn_Hani", Language.MONGOLIAN: "mon_Cyrl",
    Language.KOREAN: "kor_Hang", Language.JAPANESE: "jpn_Jpan",
    Language.HUNGARIAN: "hun_Latn", Language.FINNISH: "fin_Latn",
    "yi": "ydd_Hebr",  # yiddish in ft.
    "mt": "mlt_Latn",  # maltese in ft
    "am": "amh_Ethi",
    "en": "eng_Latn",
    "unknown": "unknown"}

# detector = LanguageDetectorBuilder.from_languages(*languages).build()
# Include only languages that are not yet extinct (= currently excludes Latin).
detector = LanguageDetectorBuilder.from_all_spoken_languages().with_preloaded_language_models().build()


def detect_lang_for_one_unit(text):
    # the unit can be anything more than 5 characters in order for lingua to succeed.
    confidence_values = detector.compute_language_confidence_values(text)
    confidence_values_highest = confidence_values[0].value
    detected_lang = confidence_values[0].language.iso_code_639_3.name.lower()  # "DEU"
    # return detected_lang
    if confidence_values_highest < 0.3:
        # if the confidence is low, use fasttext detect
        detected_lang_by_ft = ftdetect(text=text, low_memory=True)
        # {'lang':'tr', 'score':1.0}
        ft_confidence = detected_lang_by_ft["score"]
        ft_lang = detected_lang_by_ft["lang"]
        if ft_confidence >= 0.3:
            return lang_codes.get(ft_lang, ft_lang)
        else:
            return "unknown"
    else:
        return detected_lang


def load_en_words():
    dict_path = 'src/languageConfusion/words'  # downloaded from https://gist.githubusercontent.com/wchargin/8927565/raw/d9783627c731268fb2935a731a618aa8e95cf465/words

    en_words = [line.strip() for line in open(dict_path)]
    en_words = {word for word in en_words if word.islower() and len(word) > 3}
    return en_words


@functools.lru_cache(maxsize=2 ** 20)
def tokenize_text_(language: str, text: str) -> list[str]:
    """
    According to detected language, choose the corresponding tokenizer.
    """
    if language == Language.CHINESE:
        tokens = list(jieba.cut(text, cut_all=True))
    elif language == Language.ENGLISH:
        tokens = nltk.word_tokenize(text)
    elif language == Language.GERMAN:
        tokens = nltk.word_tokenize(text, language="german")
    elif language == Language.FINNISH:
        tokens = nltk.word_tokenize(text, language="finnish")
    elif language == Language.TURKISH:
        tokens = nltk.word_tokenize(text, language="turkish")
    elif language == Language.HEBREW:
        # https://github.com/YontiLevin/Hebrew-Tokenizer?tab=readme-ov-file
        tokens = [x for _, x, _, _ in hebrew_tokenizer.tokenize(text)]
    elif language == Language.ARABIC:
        tokens = tk.tokenize(text)
    elif language == Language.HINDI:
        tokens = tk.tokenize(text)
    elif language == Language.KAZAKH:
        tokens = tk.tokenize(text)
    elif language == Language.MONGOLIAN:
        tokens = tk.tokenize(text)
    elif language == Language.KOREAN:
        # https://github.com/bab2min/kiwipiepy
        tokens = [x.form for x in kiwi.tokenize(text)]
    elif language == Language.JAPANESE:
        # https://github.com/polm/fugashi?tab=readme-ov-file
        # pip install 'fugashi[unidic]'
        # python -m unidic download
        tokens = [x for x in japaneseTagger.parse(text)]
    else:
        # maltese is latin.
        tokens = nltk.word_tokenize(text)
    return tokens


def detect_language_texts(texts: list[str]):
    """
    Get language distributions at word level and line level
    """
    # get confidence.
    line_langs = []

    token_langs_counter = collections.defaultdict(int)
    # {"deu": ["welt", "wunderbar", "ausspuhren"], "eng":[ ]}
    lang2token_dict = collections.defaultdict(list)

    texts_LEN = len(texts)
    # print(f"there are {texts_LEN} texts")
    for text in texts:
        if len(text) > 0:
            line_lang_detected = detect_lang_for_one_unit(text)
            tokens = tokenize_text_(line_lang_detected, text)
            # if line_lang_detected != "unknown":
            line_langs.append(line_lang_detected)
            token_langs_line = []

            for token in tokens:
                token_lang_detected = detect_lang_for_one_unit(token)
                lang2token_dict[token_lang_detected].append(token)

                if token_lang_detected != "unknown":
                    token_langs_line.append(token_lang_detected)

            # calculate the token language ratio in each line.
            if line_lang_detected != "unknown":
                token_langs_line_counter = Counter(token_langs_line)
                # token_langs_detected_sum = sum(token_langs_line_counter.values())
                # tokens_counter += token_langs_detected_sum

                for lang_id, lang_counts in token_langs_line_counter.items():
                    token_langs_counter[lang_id] += lang_counts

    # analyze the languages detected.
    # all the lines have defined languages .
    line_langs_counter = Counter(line_langs)
    # # print(line_langs_counter)
    # if line_langs_counter.most_common(1)[0][1] == texts_LEN:
    #     print(f"all the lines have defined language {line_langs[0]}")

    # print(f"the percentage of the languages detected")
    line_langs_detected_sum = sum(line_langs_counter.values())
    line_langs_ratio_dict = {k: round(v / line_langs_detected_sum, 2) for k, v in line_langs_counter.items()}
    line_langs_ratio_dict = {k: v for k, v in line_langs_ratio_dict.items() if v > 0}
    # print(line_langs_ratio_dict)

    # for words-level languages.
    tokens_count = sum(token_langs_counter.values())
    token_langs_ratio = {x: round(k / tokens_count, 2) for x, k in token_langs_counter.items()}
    token_langs_ratio = {k: v for k, v in token_langs_ratio.items() if v > 0}
    # print(f"word level languages: {token_langs_ratio}")

    return line_langs_ratio_dict, token_langs_ratio, line_langs, lang2token_dict


def normalize(text: str) -> str:
    text = text.split('\nQ:')[0].strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("—", " ")
    text = text.replace("،", "")
    return text


def compute_language_distribution(df: pd.DataFrame, outputfolder: str, filename: str):
    # id,model,completion,task,source,language

    ids = []
    indices = []
    models = []
    comp_list = []
    tasks = []
    sources = []
    langs = []
    pred_langs = []
    lang_dist_dict = dict()
    for id, model, completion, task, source, lang in tqdm(zip(df["id"], df["model"], df["completion"],
                                                              df["task"], df["source"], df["language"])):
        try:
            completion = normalize(completion)
        except Exception:
            completion = completion
        try:
            lines = completion.split("\n")
            lines = [line for line in lines if len(line) > 5]
            if len(lines) > 0:
                pred_line_langs_ratio, pred_token_langs_ratio, pred_line_langs, pred_lang2token_dict = detect_language_texts(
                    lines)

                lang_dist_dict[f"{id}_{model}_{task}_{lang}_{source}"] = {
                    "pred_lang_line_level_ratio": pred_line_langs_ratio,
                    "pred_lang_word_level_ratio": pred_token_langs_ratio,
                    "pred_lang2token_dict": pred_lang2token_dict
                }
                for i, (line, pred_line_lang) in enumerate(zip(lines, pred_line_langs)):
                    ids.append(id)
                    indices.append(i)
                    models.append(model)
                    # each line after split
                    comp_list.append(line)
                    pred_langs.append(pred_line_lang)
                    tasks.append(task)
                    sources.append(source)
                    langs.append(lang)
        except Exception as msg:
            print(msg)

    new_df = pd.DataFrame({
        "id": ids,
        "indice": indices,
        "model": models,
        "completion": comp_list,
        "pred_lang": pred_langs,
        "task": tasks,
        "source": sources,
        "language": langs

    })
    new_df.to_csv(os.path.join(outputfolder, filename), index=False)

    with open(os.path.join(outputfolder, filename.replace(".csv", ".json")), "w") as f:
        json.dump(lang_dist_dict, f)


def main():
    datadir = "datasets/prompts_language_confusion"
    for filename in os.listdir("datasets/prompts_language_confusion"):
        if filename.endswith(".csv"):
            filepath = os.path.join(datadir, filename)
            df = pd.read_csv(filepath)
            outputfolder = "datasets/prompts_language_confusion/output"
            if not os.path.exists(os.path.join(outputfolder, filename)):
                print(f"processing file {filepath}")

                compute_language_distribution(df, outputfolder, filename)


if __name__ == '__main__':
    main()
