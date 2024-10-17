import matplotlib.pyplot as plt
import seaborn as sns


def plot_lc_inversion(df, by_entropy, outputfile, lang, metric_name="BLEU", generation_setting="crosslingual", level="line"):
    # Assuming df_eval_lang_entropy_melted contains columns: 'language', 'entropy', 'step', and 'f1_score'

    # Create the figure and the first axis (for entropy)
    fig, ax1 = plt.subplots(figsize=(26, 7))

    # print(lang)
    # print(df)
    # Plot the barplot for entropy on the first axis
    if lang=="train":
        p = sns.barplot(x='lang', y='entropy', data=df, hue="step", palette="colorblind", ax=ax1)
    else:
        p = sns.barplot(x='language', y='entropy', data=df, hue="step", palette="colorblind", ax=ax1)

    # Add hatches to the bars
    # hatches = '/'
    # step_with_hatch = "Step1"
    # for patch, (step) in zip(p.patches, df['step']):
    #     if step == step_with_hatch:
    #         patch.set_hatch(hatches)
    #     if step == "Step50+sbeam8":
    #         patch.set_hatch('\\')

    # Customize the first y-axis (for entropy)

    if by_entropy=="weighted_entropy":
        ylabel = "Weighted Entropy"
    elif by_entropy=="entropy_out":
        ylabel = "Entropy[OUT]"
    elif by_entropy=="entropy_all":
        ylabel = "Entropy[ALL]"

    ax1.set_ylabel(ylabel, fontsize=26)
    ax1.set_xlabel('Language', fontsize=26)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    plt.xticks(rotation=30)


    # Plot the line plot for F1 score on the second y-axis
    # if lang == "eval":
        # Create a second y-axis (for F1 score)
    ax2 = ax1.twinx()  # This creates the second y-axis sharing the same x-axis

    dashes = [(1, 0), (5, 5), (3, 1, 1, 1)]  # solid, dashed, dotted

    if lang == "train":
        sns.lineplot(x='lang', y='inversion_avg', data=df, hue="step",
                     marker="o", linewidth=2, markersize=10, palette="colorblind", ax=ax2, style="step",
                     dashes=dashes)
    else:
        sns.lineplot(x='language', y='inversion_avg', data=df, hue="step",
                     marker="o", linewidth=2, markersize=10, palette="colorblind", ax=ax2,  style="step",
                     dashes=dashes)


    # sns.lineplot(x='language', y='mt5', data=df_eval_lang_entropy_melted_f1_bleu,
    #              marker="o", linewidth=2, markersize=10, palette="Set2", ax=ax1,)

    # Customi

    # Customize the second y-axis (for F1 score)
    ax2.set_ylabel(metric_name.capitalize(), fontsize=26)
    ax2.tick_params(axis='y', labelsize=20)

    # Add the title
    plt.title(f'{generation_setting.capitalize()} Setting at {level.capitalize()} Level', fontsize=30)

    # Add the legend for the steps (can be adjusted for both y-axes)
    ax1.legend(title="Steps", fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.2)
    if lang == "eval":
        ax2.legend(title=metric_name.capitalize(), fontsize=14, title_fontsize=16, loc='upper right', bbox_to_anchor=(1.16, 0.7), borderaxespad=0.2)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(outputfile, format='pdf', bbox_inches='tight')

    # Show the plot
    # plt.show()
