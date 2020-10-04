import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
import numpy as np
import os


def bar_plots(output_path, data):
    """

    :param output_path:
    :param data:
    :return:
    """
    protocol_labels = data[0]
    experiment_labels = data[1]
    metric_labels = data[2]
    x = np.arange(len(protocol_labels))  # the label locations

    colors = ['blue', 'orange', 'green']

    for metric_label in data[3]:
        metric_scores = data[3][metric_label]
        num_groups = len(metric_scores)
        width = 0.7 / num_groups  # the width of the bars
        fig, ax = plt.subplots()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)

        for index, experiment_label in enumerate(metric_scores):
            experiment_scores_mean, experiment_scores_err = metric_scores[experiment_label]
            rects = ax.bar(x - (num_groups - 1) * width / 2 + width * index, experiment_scores_mean, width,
                           label=experiment_labels[index],
                           color=colors[index])
            autolabel(rects)

        ax.set_ylabel('fractional success')
        ax.set_title(metric_label[5:])
        ax.set_xticks(x)
        ax.set_ylim((0, 1.2))
        ax.set_xticklabels(protocol_labels, rotation='vertical')
        ax.legend()

        fig.tight_layout()

        plt.savefig(
            os.path.join(output_path, 'bar_plots_{}.png'.format(metric_label)))


def bar_plots_with_protocol_table(output_path, data, protocol_settings, task):
    """

    :param output_path:
    :param data:
    :param protocol_settings:
    :param task:
    :return:
    """
    protocol_labels = data[0]
    protocol_ids = ['P{}'.format(i) for i in range(len(protocol_labels))]
    experiment_labels = data[1]
    metric_labels = data[2]
    x = np.arange(len(protocol_labels))  # the label locations
    mpl.rc('text', usetex=True)
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ['computer modern'],
        "axes.labelsize": 10,
        "font.size": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    mpl.rcParams.update(tex_fonts)
    colors = ["#3182bd",
              "#de2d26",
              "#31a354",
              "#9ecae1",
              "#fc9272",
              "#a1d99b",
              "#deebf7",
              "#fee0d2",
              "#e5f5e0"]

    colors = ["#3182bd",
              "#9ecae1",
              "#deebf7",
              "#de2d26",
              "#fc9272",
              "#fee0d2",
              "#31a354",
              "#a1d99b",
              "#e5f5e0"]

    for metric_label in data[3]:
        metric_scores = data[3][metric_label]
        num_groups = len(metric_scores)
        fig_width = 5.5
        fig_height = fig_width / 3
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.set_size_inches(fig_width, fig_height)
        width = 0.7 / num_groups  # the width of the bars
        spare_width = 0.5
        ax.set_xlim(-spare_width, len(protocol_labels) - spare_width)
        row_labels = list(protocol_settings[list(protocol_settings.keys())[0]].keys())
        for index, experiment_label in enumerate(metric_scores):
            experiment_scores_mean, experiment_scores_err = metric_scores[experiment_label]
            experiment_scores_std_list_upper = [min(std, 1.0 - mean) for mean, std in zip(experiment_scores_mean,
                                                                                          experiment_scores_err)]
            error_kw = dict(lw=5, capsize=5, capthick=3)
            plt.bar(x - (num_groups - 1) * width / 2 + width * index, experiment_scores_mean, width,
                    yerr=(experiment_scores_err, experiment_scores_std_list_upper),
                    error_kw=dict(lw=1, capsize=1, capthick=1),
                    label=experiment_labels[index],
                    color=colors[index])

        cell_text = list()
        for row_label in row_labels:
            cell_text.append(['{}'.format(protocol_settings[experiment_label][row_label]) for
                              experiment_label in list(protocol_settings.keys())])

        ax.set_ylabel('fractional success', fontsize=8)
        # ax.set_title(task + '  ' + metric_label[5:].replace('_', ' '))
        plt.legend(ncol=3, loc='upper right', prop={'size': 6})
        ax.set_ylim((0, 1.2))
        plt.yticks(fontsize=8)
        ax.get_xaxis().set_visible(False)
        table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=protocol_ids,
                          loc='bottom')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        cellDict = table.get_celld()
        for i in range(-1, len(protocol_ids)):
            if i != -1:
                cellDict[(0, i)].set_height(.11)
            for j in range(1, len(row_labels) + 1):
                if j == 2:
                    cellDict[(j, i)].set_height(.15)
                else:
                    cellDict[(j, i)].set_height(.11)
                cellDict[(j, i)].set_fontsize(6)
        fig.subplots_adjust(bottom=0.33, left=0.11, right=0.99, top=0.98)

        plt.savefig(
            os.path.join(output_path, 'bar_plots_protocol_table_{}.pdf'.format(metric_label)), dpi=300)
