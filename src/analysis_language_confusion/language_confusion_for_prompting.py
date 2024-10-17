import pandas as pd
import numpy as np

from ast import literal_eval

from scipy.stats import entropy

code2lang = {
    'ar': 'ara', 'de': 'deu', 'en': 'eng', 'es': 'spa', 'fr': 'fra', 'hi': 'hin',
    'id': 'ind', 'it': 'ita', 'ja': 'jpn', 'ko': 'kor', 'pt': 'por', 'ru': 'rus',
    'tr': 'tur', 'vi': 'vie', 'zh': 'zho'}
iso3names = {
    'ara': 'Arabic', 'deu': 'German', 'eng': 'English', 'spa': 'Spanish',
    'fra': 'French', 'hin': 'Hindi', 'ind': 'Indonesian', 'ita': 'Italian',
    'jpn': 'Japanese', 'kor': 'Korean', 'por': 'Portuguese', 'rus': 'Russian',
    'tur': 'Turkish', 'vie': 'Vietnamese', 'zho': 'Chinese'
}


def get_entropy_for_all(x):
    try:
        return entropy(np.array(list(x.values())))
    except Exception:
        return np.nan


def get_entropy_outside_eval(row, level):
    """Get entropy outside eval/target langauges."""
    try:
        pred_langs_dist = row[f"{level}_level_pred_langs_dist"]
        eval_lang = row["lang"]
        dist_others = []
        for lang, dist in pred_langs_dist.items():
            if lang != eval_lang:
                dist_others.append(dist)
        return entropy(np.array(dist_others))
    except Exception:
        return np.nan

def get_reweighted_entropy(row, level):
    try:
        "Re-weighted entropy"
        pred_langs_dist = row[f"{level}_level_pred_langs_dist"]
        eval_lang = row["lang"]
        reweighted_entropy = 0
        for lang, prob in pred_langs_dist.items():
            if lang == eval_lang:
                reweighted_entropy -= (1 - prob) * np.log(prob)
            else:
                reweighted_entropy -= prob * np.log(prob)
        return reweighted_entropy
    except Exception:
        return np.nan


def safe_literal_eval(input_string):
    # Replace 'NaN' with np.nan or float('nan') in the string
    # input_string = input_string.replace('NaN', 'np.nan')

    # Safely evaluate the expression
    try:
        return literal_eval(input_string)
    except (ValueError, SyntaxError):
        return np.nan



def processing_langauge_confusion_dataframe(by_entropy="reweighted_entropy"):
    df = pd.read_csv(f"datasets/prompts_language_confusion/lang2dist/all.csv")
    # give language names
    df["lang"] = df["lang"].map(code2lang)
    # df = df.dropna(subset=["line_level_pred_langs", "word_level_pred_langs"])

    df["line_level_pred_langs"] = df["line_level_pred_langs"].apply(safe_literal_eval)
    df["word_level_pred_langs"] = df["word_level_pred_langs"].apply(safe_literal_eval)

    def prob_dist_norm(x):
        try:
            x_dict = {k: v[0] for k, v in x.items()}
            unk_value = round(1 - sum(x_dict.values()), 2)
            if unk_value > 0:
                x_dict["unk"] = unk_value
            return x_dict
        except Exception as msg:
            return x

    df["line_level_pred_langs_dist"] = df["line_level_pred_langs"].apply(prob_dist_norm)
    df["word_level_pred_langs_dist"] = df["word_level_pred_langs"].apply(prob_dist_norm)

    # monolingual or crosslingual
    # df = df[df["task"] == generation_setting]

    ## calculate entropies.
    # get entropy for all languages in pred langs on normalzied distributions
    if by_entropy == "entropy_all":
        df[f"line_level_{by_entropy}"] = df["line_level_pred_langs_dist"].apply(get_entropy_for_all)
        df[f"word_level_{by_entropy}"] = df["word_level_pred_langs_dist"].apply(get_entropy_for_all)
    # get entropy for all languages that are not train or eval languages
    elif by_entropy == "entropy_out":
        df[f"line_level_{by_entropy}"] = df.apply(get_entropy_outside_eval, axis=1, args=("line",))
        df[f"word_level_{by_entropy}"] = df.apply(get_entropy_outside_eval, axis=1, args=("word",))
    # modified entropy
    elif by_entropy == "reweighted_entropy":
        df[f"line_level_{by_entropy}"] = df.apply(get_reweighted_entropy, axis=1, args=("line",))
        df[f"word_level_{by_entropy}"] = df.apply(get_reweighted_entropy, axis=1, args=("word",))

    df.to_csv(f"results/prompting_language_confusion/dataframes/all_{by_entropy}_entropy.csv")
    # look at dataframe.
    print(df.head(2))
    llm_dict = {
        'command-r-base': "Command R base", 'command-r': "Command R",
        'command-r-plus-base': "Command R+ base", 'command-r-plus': "Command R+",
        'gpt-3.5-turbo': "GPT-3.5 Turbo", 'gpt-4-turbo': "GPT-4 Turbo",
        'mistral-large': "Mistral Large", 'mistral-8x7b': "Mistral 8x7B",
        'llama-2-instruct': "Llama 2 70B-I", 'llama-3-instruct': "Llama 3 70B-I"}

    df["LLM_name"] = df["LLM"].map(llm_dict)
    df_line = df.groupby(['LLM', 'source', "lang", "task"], as_index=False).agg(
        avg_line_weighted_entropy=(f"line_level_{by_entropy}", 'mean'))
    df_word = df.groupby(['LLM', 'source', "lang", "task"], as_index=False).agg(
        avg_word_weighted_entropy=(f"word_level_{by_entropy}", 'mean'))
    df_both = pd.merge(df_line, df_word, on=["LLM", "source", "lang", "task"])

    assert len(df_word) == len(df_line) == len(df_both)

    print(df_line.head(2))
    print(df_word.head(2))
    # calculate the LPR and WPR for language confusion
    df_results = pd.read_csv("datasets/prompts_language_confusion/results/reproduction_pass_rates_results.csv")
    df_results["lang"] = df_results["lang"].map(code2lang)
    df_results.rename(columns={"model": "LLM"}, inplace=True)

    df_merge = pd.merge(df_both, df_results, on=["LLM", "source", "lang", "task"], how="left")
    df_merge.to_csv(f"datasets/prompts_language_confusion/results/{by_entropy}_passrates.csv", index=False)


if __name__ == '__main__':
    import plac

    plac.call(processing_langauge_confusion_dataframe)
