import os
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
from ftlangdetect import detect as ftdetect
import collections
import numpy as np
from collections import Counter
import jieba
import nltk
import json

import hebrew_tokenizer
from fugashi import Tagger

japaneseTagger = Tagger("-Owakati")
tk = nltk.WordPunctTokenizer()
from typing import List, Union
import scipy

from kiwipiepy import Kiwi

kiwi = Kiwi()

# load language codes from iso2 to iso3.
with open("language_confusion/letter_codes.json") as f:
    lang_codes = json.load(f)

texts = ["אלעס ארום די G-7 סאמיט צוזאמפאל - בלאט 8 - אידישע וועלט פארומס תגובהדורך כא",
         "⚽⚽🌙【备用网址yabovp.com】vp.com.tw【每個人都希望能贏得這麼多錢,",
         "“I’m sure you’re excited! I’m sure you’re excited! I’m sure you’re excited!",
         "BIL-FILMAT: L-Ispeaker ma jilqax it-talba ta' Delia minħabba biża",
         '自然言語処理を勉強しています',
         " فستان زفاف أنيق بقصّة الأميرة من نسيج الميكادو، مُزيّن بالأزهار",
         " नई दिल्ली, प्रधानमंत्री नरेंद्र मोदी का कार्यकाल पूरा होने में केवल 14 महीने का वक्त बा",
         "Ыбырай Алтынсарин - Ы - Ұлы заман Тұлғалары - Скачать Рефераты",
         "Біздің өміріміз үлкен өзен іспетті. Сіздің қайығыңыздың қиындықтардан жеңіл өтіп, махаббат иірімінде басқаруын жоғалтпай, бақыт сарқырамасына жетуін тілеймін!",
         " Google Текстийн зарын өөрчлөлтийг анхаарч үзэх 3 зүйл | Martech Zone Google Текстийн зарын",
         "'અમારા વિરોધીને લગ્નમાં કેમ બોલાવ્યો?' કહી કન્યાના મા-બા",
         " سوناکشی سنہا سوشل میڈیا میمز کے نشے میں مبتلا ہو گئیں اداکارہ نہ صرف خود",
         " ਕਰੋਨਾ ਮਹਾਂਮਾਰੀ ਦੇ ਚੱਲਦੇ ਫੋਟੋਗ੍ਰਾਫਰ ਦੀਆਂ ਬੰਦ ਪਈਆਂ ਦੁਕਾਨਾਂ ਖੋ",
         " 문의 주신 제품은 재입고 예정은 있으나 현재 정확한 일정은 확인 중인 점 양해"
         ]

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


def tokenize_text_(language, text):
    """
    According to detected language, choose the corresponding tokenizer.
    """
    if language == Language.CHINESE:
        return list(jieba.cut(text, cut_all=True))
    elif language == Language.ENGLISH:
        return nltk.word_tokenize(text)
    elif language == Language.GERMAN:
        return nltk.word_tokenize(text, language="german")
    elif language == Language.FINNISH:
        return nltk.word_tokenize(text, language="finnish")
    elif language == Language.TURKISH:
        return nltk.word_tokenize(text, language="turkish")
    elif language == Language.HEBREW:
        # https://github.com/YontiLevin/Hebrew-Tokenizer?tab=readme-ov-file
        return [x for _, x, _, _ in hebrew_tokenizer.tokenize(text)]
    elif language == Language.ARABIC:
        return tk.tokenize(text)
    elif language == Language.HINDI:
        return tk.tokenize(text)
    elif language == Language.KAZAKH:
        return tk.tokenize(text)
    elif language == Language.MONGOLIAN:
        return tk.tokenize(text)
    elif language == Language.KOREAN:
        # https://github.com/bab2min/kiwipiepy
        return [x.form for x in kiwi.tokenize(text)]
    elif language == Language.JAPANESE:
        # https://github.com/polm/fugashi?tab=readme-ov-file
        # pip install 'fugashi[unidic]'
        # python -m unidic download
        return [x for x in japaneseTagger.parse(text)]
    else:
        # maltese is latin.
        return nltk.word_tokenize(text)


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def text_comparison_metrics(preds, references):
    num_preds = len(preds)
    precision_sum = 0.0
    recall_sum = 0.0
    num_overlapping_words = []
    num_overlapping_bigrams = []
    num_overlapping_trigrams = []
    num_true_words = []
    num_pred_words = []
    f1s = []

    preds_langs_line_level = []
    references_langs_line_level = []

    for i in range(num_preds):
        reference_lang = detect_lang_for_one_unit(references[i])
        pred_lang = detect_lang_for_one_unit(preds[i])

        preds_langs_line_level.append(pred_lang)
        references_langs_line_level.append(reference_lang)

        # tokenize text using reference language.
        true_words = tokenize_text_(reference_lang, references[i])
        pred_words = tokenize_text_(pred_lang, preds[i])

        num_true_words.append(len(true_words))
        num_pred_words.append(len(pred_words))
        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)
        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)
        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

        precision_sum += precision
        recall_sum += recall
        num_overlapping_words.append(
            count_overlapping_ngrams(true_words, pred_words, 1)
        )
        num_overlapping_bigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 2)
        )
        num_overlapping_trigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 3)
        )
    set_token_metrics = {
        "token_set_precision": (precision_sum / num_preds),
        "token_set_recall": (recall_sum / num_preds),
        "token_set_f1": mean(f1s),
        "token_set_f1_sem": sem(f1s),
        "n_ngrams_match_1": mean(num_overlapping_words),
        "n_ngrams_match_2": mean(num_overlapping_bigrams),
        "n_ngrams_match_3": mean(num_overlapping_trigrams),
        "num_true_words": mean(num_true_words),
        "num_pred_words": mean(num_pred_words),
    }

    return set_token_metrics, f1s, preds_langs_line_level, references_langs_line_level


def eval_for_sequences_token_metrics(
        filepath, outputfolder):
    df = pd.read_csv(filepath)
    print(df.head())
    try:
        preds = [x.replace("query: ", "") for x in df["pred"].tolist()]
        references = [x.replace("query: ", "") for x in df["labels"].tolist()]
        set_token_metrics, f1s, preds_langs_line_level, references_langs_line_level = \
            text_comparison_metrics(preds, references)

        with open(os.path.join(outputfolder, "set_token_eval_all_langs_0.3.json"), "w") as f:
            json.dump(set_token_metrics, f)

        df["preds_lang"] = preds_langs_line_level
        df["labels_lang"] = references_langs_line_level
        df["token_set_f1"] = f1s
        df.to_csv(os.path.join(outputfolder, "set_token_eval_all_langs_0.3.csv"), index=False)

    except Exception as msg:
        print(f"Exception {msg}")


# monolingual embeddings, decode nonsense.
# multilingual embeddings, decode sensible
# should always look after the labels.
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
    print(f"there are {texts_LEN} texts")
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
    # print(line_langs_counter)
    if line_langs_counter.most_common(1)[0][1] == texts_LEN:
        print(f"all the lines have defined language {line_langs[0]}")

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


def processing_filepath_lang(filepath, outputfolder):
    df = pd.read_csv(filepath)
    print(df.head())
    try:
        preds = [x.replace("query: ", "") for x in df["pred"].tolist()]
        references = [x.replace("query: ", "") for x in df["labels"].tolist()]

        print("************* for predictions *************")
        pred_line_langs_ratio, pred_token_langs_ratio, pred_line_langs, pred_lang2token_dict = detect_language_texts(preds)

        print("************* for references *************")
        refer_line_langs_ratio, refer_token_langs_ratio, refer_line_langs, refer_lang2token_dict = detect_language_texts(references)
        lang_info = {
            "pred_lang_line_level_ratio": pred_line_langs_ratio,
            "pred_lang_word_level_ratio": pred_token_langs_ratio,
            "labels_lang_line_level_ratio": refer_line_langs_ratio,
            "labels_lang_word_level_ratio": refer_token_langs_ratio
        }
        df["pred_lang"] = pred_line_langs
        df["labels_lang"] = refer_line_langs
        df.to_csv(os.path.join(outputfolder, "eval_lang_all_langs_0.3.csv"), index=False)

        print(lang_info)

        with open(os.path.join(outputfolder, "eval_lang_all_langs_0.3.json"), "w") as f:
            json.dump(lang_info, f)

        # predicted lang2token dictionary
        with open(os.path.join(outputfolder, "pred_all_lang2token_0.3.json"), "w") as f:
            json.dump(pred_lang2token_dict, f)

        # label lang2token dictionary
        with open(os.path.join(outputfolder, "label_all_lang2token_0.3.json"), "w") as f:
            json.dump(refer_lang2token_dict, f)

    except Exception as msg:
        print(f"exception: {msg}")


def language_detector_eval_datasets_batch(lingual="multilingual", inversion="inverter"):
    folderpath = f"eval_logs/{lingual}"
    for file in os.listdir(folderpath):
        filepath = os.path.join(folderpath, file)
        # eval_logs/multilingual/eval_mt5_me5_arb_Arab_32_2layers_inverter.json
        if file.endswith(".json"):
            if inversion == "inverter" and inversion in file:
                print(f"processing and detect languages {filepath}")
                with open(filepath, "r") as f:
                    eval_logs = json.load(f)
                # "model": "yiyic/mt5_me5_arb_Arab_32_2layers_inverter",
                if eval_logs["model"] is not None:
                    model_outputfolder = os.path.join("saves", eval_logs["model"].replace("/", "__"))
                    if os.path.exists(model_outputfolder):
                        print(f"{model_outputfolder} exists...")
                        for eval in eval_logs["evaluations"]:
                            print(f"processing eval dataset {eval['dataset']}")
                            # saves/yiyic__mt5_me5_arb_Arab_32_2layers_inverter/decoded_eval_1721930770
                            decoded_file_folder = eval["embeddings_file"]
                            decoded_file = eval["output_file"].replace(" ", "")
                            # outputfile = os.path.join(decoded_file_folder, "eval_lang_all_langs.csv")
                            outputfile = os.path.join(decoded_file_folder, "pred_all_lang2token_0.3.json")
                            if not os.path.exists(outputfile):
                                print(f"processing {outputfile}")
                                processing_filepath_lang(decoded_file, decoded_file_folder)

                            set_token_outputfile = os.path.join(decoded_file_folder, "set_token_eval_all_langs_0.3.csv")
                            # if not os.path.exists(set_token_outputfile):
                            #     print(f"processing set_token_metrics")
                            #     eval_for_sequences_token_metrics(decoded_file, decoded_file_folder)

            elif "corrector" in file and inversion in file:
                if "_semitic-fami_" not in file:
                    print(f"processing and detect languages {filepath}")
                    with open(filepath, "r") as f:
                        eval_logs = json.load(f)
                    if eval_logs["model"] is not None:
                        model_outputfolder = os.path.join("saves", eval_logs["model"].replace("/", "__"))
                        if os.path.exists(model_outputfolder):
                            for eval_dataset, eval_steps_results in eval_logs["evaluations"].items():
                                print(f"processing eval_dataset {eval_dataset}")
                                for step, step_results in eval_steps_results.items():
                                    if step.endswith("steps 1"):
                                        if "embeddings_files" in step_results:
                                            decoded_file_folder = step_results["embeddings_files"]
                                            decoded_file = step_results["output_files"].replace(" ", "")
                                            # outputfile = os.path.join(decoded_file_folder, "eval_lang_all_langs.csv")
                                            outputfile = os.path.join(decoded_file_folder, "pred_all_lang2token_0.3.json")
                                            if not os.path.exists(outputfile):
                                                print(f"processing {outputfile}")
                                                processing_filepath_lang(decoded_file, decoded_file_folder)

                                            set_token_outputfile = os.path.join(decoded_file_folder,
                                                                                "set_token_eval_all_langs_0.3.csv")
                                            # if not os.path.exists(set_token_outputfile):
                                            #     print(f"processing set_token_metrics")
                                            #
                                            #     eval_for_sequences_token_metrics(decoded_file, decoded_file_folder)

                                    if step.endswith("beam width 8"):
                                        if "embeddings_files" in step_results:
                                            decoded_file_folder = step_results["embeddings_files"]
                                            decoded_file = step_results["output_files"].replace(" ", "")
                                            # outputfile = os.path.join(decoded_file_folder, "eval_lang_all_langs.csv")
                                            outputfile = os.path.join(decoded_file_folder, "pred_all_lang2token_0.3.json")
                                            if not os.path.exists(outputfile):
                                                print(f"processing {outputfile}")
                                                processing_filepath_lang(decoded_file, decoded_file_folder)

                                            set_token_outputfile = os.path.join(decoded_file_folder,
                                                                                "set_token_eval_all_langs_0.3.csv")

                                            # if not os.path.exists(set_token_outputfile):
                                            #     print(f"processing set_token_metrics")
                                            #     eval_for_sequences_token_metrics(decoded_file, decoded_file_folder)

                        else:
                            print(f"{model_outputfolder} does not exist")


if __name__ == '__main__':
    import plac

    plac.call(language_detector_eval_datasets_batch)

