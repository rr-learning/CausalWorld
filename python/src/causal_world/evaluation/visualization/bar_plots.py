import matplotlib.pyplot as plt
import numpy as np
import os


def bar_plots(output_path, data):
    protocol_labels = data[0]
    experiment_labels = data[1]
    metric_labels = data[2]
    x = np.arange(len(protocol_labels))  # the label locations

    colors = ['blue', 'orange', 'green']

    for (metric_label, metric_scores) in data[3]:
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

        for index, experiment_scores in enumerate(metric_scores):
            experiment_label, experiment_list = experiment_scores
            rects = ax.bar(x - (num_groups - 1) * width / 2 + width * index, experiment_list, width,
                           label=experiment_labels[index],
                           color=colors[index])
            autolabel(rects)

        ax.set_ylabel('fractional success')
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_ylim((0, 1.2))
        ax.set_xticklabels(protocol_labels, rotation='vertical')
        ax.legend()

        fig.tight_layout()

        plt.savefig(
            os.path.join(output_path, 'bar_plots_{}.png'.format(metric_label)))


def bar_plots_with_table(output_path, data):
    protocol_labels = data[0]
    experiment_labels = data[1]
    metric_labels = data[2]
    x = np.arange(len(protocol_labels))  # the label locations

    colors = ['blue', 'orange', 'green']

    for (metric_label, metric_scores) in data[3]:
        num_groups = len(metric_scores)
        width = 0.7 / num_groups  # the width of the bars
        fig, ax = plt.subplots()
        spare_width = 0.5
        ax.set_xlim(-spare_width, len(protocol_labels) - spare_width)
        cell_text = list()
        row_labels = list()
        for index, experiment_scores in enumerate(metric_scores):
            experiment_label, experiment_list = experiment_scores
            plt.bar(x - (num_groups - 1) * width / 2 + width * index, experiment_list, width,
                    label=experiment_labels[index],
                    color=colors[index])
            cell_text.append(['{}'.format(round(x, 2)) for x in experiment_list])
            row_labels.append(experiment_label)

        ax.set_ylabel('fractional success')
        ax.set_title(metric_label)
        #ax.set_xticks(x)
        ax.set_ylim((0, 1.2))
        ax.get_xaxis().set_visible(False)
        #ax.set_xticklabels(protocol_labels, rotation='vertical')
        table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=[str(i) for i in range(len(protocol_labels))],
                          rowColours=colors,
                          loc='bottom')
        fig.subplots_adjust(bottom=0.1)
        fig.tight_layout()

        plt.savefig(
            os.path.join(output_path, 'bar_plots_table_{}.png'.format(metric_label)))
