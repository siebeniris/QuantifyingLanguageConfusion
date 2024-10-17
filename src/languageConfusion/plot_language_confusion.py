import pandas as pd
import json
import os
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from collections import defaultdict
from ast import literal_eval

cmap = get_cmap('Pastel2', 22)
colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(22)]

# 20 languages.
labels = ['amh_Ethi',
          'ydd_Hebr',
          'mhr_Cyrl',
          'tur_Latn',
          'hun_Latn',
          'kor_Hang',
          'pan_Guru',
          'deu_Latn',
          'jpn_Jpan',
          'kaz_Cyrl',
          'mon_Cyrl',
          'hin_Deva',
          'etc.',  # others
          'mlt_Latn',
          'fin_Latn',
          'guj_Gujr',
          'urd_Arab',
          'heb_Hebr',
          'arb_Arab',
          'cmn_Hani',
          'sin_Sinh']

color_map = {label: colors[i] for i, label in enumerate(labels)}
sns.set_palette("pastel")


def load_data_level_lingual(filepath):
    # language_confusion/langdist_data/dataset2langdist_line_level_multi.csv
    df = pd.read_csv(filepath)
    d2l = defaultdict(dict)

    for model, eval_lang, step, pred_langs in zip(df["model"], df["eval_lang"], df["step"], df["pred_langs"]):
        if model not in d2l:
            d2l[model] = dict()
        if eval_lang not in d2l[model]:
            d2l[model][eval_lang] = dict()

        if step not in d2l[model][eval_lang]:
            d2l[model][eval_lang][step] = literal_eval(pred_langs)
    return d2l


def plot_language_confusion_for_one_dataset(d, model_name, dataset, level, output_folder):
    df_dataset = pd.DataFrame.from_records(d[model_name][dataset]).T
    df_dataset = df_dataset.fillna(0)
    print(df_dataset)
    df_dataset["etc."] = 1 - df_dataset.sum(axis=1)
    df_dataset = df_dataset.rename(index={"Labels": "Target", "Step50+sbeam8": "Step50(8)"})

    df_dataset = df_dataset.reindex(index=["Step50(8)", "Step1", "Base", "Target"])
    df_dataset["Step"] = df_dataset.index

    print(df_dataset)

    ax = df_dataset.plot(
        x="Step",
        kind="barh",
        stacked=True,
        figsize=(4.2, 2),
        grid=True,
        mark_right=True,
        color=[color_map.get(label, '#333333') for label in df_dataset.columns if label != 'Step']
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    # Add hatches to the bars
    hatches = ['//', '\\\\', '//', '\\\\', '//', '\\\\', '//', '\\\\', '//',
               '\\\\', '//', '\\\\', '//', '\\\\', '//', '\\\\', '//', '\\\\', '//',
               '\\\\', '//', '\\\\', '//', '\\\\', '//', '\\\\', '//']

    num_hatches = len(hatches)
    num_labels = len(df_dataset.columns) - 1  # excluding the 'Step' column

    edge_color = 'darkgray'
    for j, bar_container in enumerate(ax.containers):
        hatch = hatches[j % num_hatches]
        for patch in bar_container.get_children():
            patch.set_hatch(hatch)
            patch.set_edgecolor(edge_color)
            # patch.set_facecolor('white')  # Set face color to white to see hatch color clearly
            patch.set_linewidth(0.5)

    # Update legend with hatches
    handles, labels = ax.get_legend_handles_labels()
    for handle, hatch in zip(handles, hatches[:num_labels]):
        for patch in handle.get_children():
            patch.set_hatch(hatch)
            patch.set_edgecolor(edge_color)
            # patch.set_facecolor('white')  # Set face color to white to see hatch color clearly
            patch.set_linewidth(0.5)

    legend = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylabel("")
    plt.xlabel(f"Per. Lang. at {level} Level")
    plt.title(F"{model_name} : {dataset}", fontsize=10, loc="left")
    plt.tight_layout()

    plt.savefig(f'{output_folder}/{model_name}_{dataset}.pdf',
                dpi=300,
                bbox_inches='tight', bbox_extra_artists=[legend])
    plt.close()


def main():
    for mode in ["multi", "mono", "mono+multi"]:
        for level in ["line_level", "word_level"]:
            # for level in [ "word_level"]:
            langdist_df = f"language_confusion/langdist_data/dataset2langdist_{level}_{mode}.csv"
            print(f"Plotting {level} and {mode} language confusion.")
            d2l = load_data_level_lingual(langdist_df)
            outputdir = f"results/plots/{level}/{mode}"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            for model, dataset in d2l.items():
                for dataset, v in dataset.items():
                    # labels.base, step1, step50.
                    if len(v) == 4:
                        if level == "line_level":
                            plot_language_confusion_for_one_dataset(d2l, model, dataset, "Line", outputdir)
                        elif level == "word_level":
                            plot_language_confusion_for_one_dataset(d2l, model, dataset, "Word", outputdir)
            print("*" * 20)


if __name__ == '__main__':
    main()
