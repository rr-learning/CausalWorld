import matplotlib.pyplot as plt
import numpy as np
import os


def bar_plots(output_path, data):
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
    protocol_labels = data[0]
    protocol_ids = ['P{}'.format(i) for i in range(len(protocol_labels))]
    experiment_labels = data[1]
    metric_labels = data[2]
    x = np.arange(len(protocol_labels))  # the label locations

    colors = ["#a6cee3",
              "#1f78b4",
              "#b2df8a",
              "#33a02c",
              "#fb9a99",
              "#e31a1c",
              "#fdbf6f",
              "#ff7f00",
              "#cab2d6"]
    for metric_label in data[3]:
        metric_scores = data[3][metric_label]
        num_groups = len(metric_scores)
        width = 0.7 / num_groups  # the width of the bars
        fig, ax = plt.subplots(figsize=(12, 4))
        spare_width = 0.5
        ax.set_xlim(-spare_width, len(protocol_labels) - spare_width)
        row_labels = list(protocol_settings[list(protocol_settings.keys())[0]].keys())
        for index, experiment_label in enumerate(metric_scores):
            experiment_scores_mean, experiment_scores_err = metric_scores[experiment_label]
            experiment_scores_std_list_upper = [min(std, 1.0 - mean) for mean, std in zip(experiment_scores_mean,
                                                                                          experiment_scores_err)]
            plt.bar(x - (num_groups - 1) * width / 2 + width * index, experiment_scores_mean, width,
                    yerr=(experiment_scores_err, experiment_scores_std_list_upper), capsize=2,
                    label=experiment_labels[index],
                    color=colors[index])

        cell_text = list()
        for row_label in row_labels:
            cell_text.append(['{}'.format(protocol_settings[experiment_label][row_label]) for
                              experiment_label in list(protocol_settings.keys())])

        ax.set_ylabel('fractional success', fontsize=11)
        ax.set_title(task + '  ' + metric_label[5:])
        #ax.set_xticks(x)
        plt.legend(ncol=9, loc='upper right')
        ax.set_ylim((0, 1.2))
        plt.yticks(fontsize=11)
        ax.get_xaxis().set_visible(False)
        #ax.set_xticklabels(protocol_labels, rotation='vertical')
        table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=protocol_ids,
                          loc='bottom')
        #table.set_fontsize(28)
        #table.scale(1.5, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        cellDict = table.get_celld()
        for i in range(-1, len(protocol_ids)):
            if i != -1:
                cellDict[(0, i)].set_height(.1)
            for j in range(1, len(row_labels) + 1):
                cellDict[(j, i)].set_height(.1)
                cellDict[(j, i)].set_fontsize(9)
        fig.subplots_adjust(bottom=0.25)
        fig.tight_layout()

        plt.savefig(
            os.path.join(output_path, 'bar_plots_protocol_table_{}.pdf'.format(metric_label)))
