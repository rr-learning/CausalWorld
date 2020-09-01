
def aggregated_data_from_experiments(experiments, contains_err=False):
    """
    experiments: Is a dict of score dicts with each key being the scores of an experiments
    contains_err: If True, for each metric score a error is expected under the key metric_label + '_std'


    Returns: a structured list that can be processed by the plotters
    first element is a list of the protocol_names
    second element is a list of the experiment_labels
    third element is a list of the metric_labels
    fourth element is the dict the scores:
        per metric label (key) the value are the metric_scores (is a dict)
            per experiment_label (key) the value is a tuple containing two lists:
                experiment_scores_list is a list of scores for each protocol
                experiment_scores_std_list is a list of stds of the scores for each protocol
    """

    experiment_labels = list(experiments.keys())
    protocol_labels = list(experiments[list(
        experiments.keys())[0]].keys())
    metric_labels = []
    for label in experiments[experiment_labels[0]][protocol_labels[0]].keys():
        if 'mean' in label and 'std' not in label:
            metric_labels.append(label)

    data = list()
    data.append(protocol_labels)
    data.append(experiment_labels)
    data.append(metric_labels)
    scores = dict()
    for metric_label in metric_labels:
        metric_scores = dict()
        for experiment_label in experiment_labels:
            experiment_scores_list = list()
            experiment_scores_std_list = list()
            for evaluation_protocol in protocol_labels:
                experiment_scores_list.append(
                    experiments[experiment_label][evaluation_protocol][metric_label])
                if contains_err:
                    experiment_scores_std_list.append(
                        experiments[experiment_label][evaluation_protocol][metric_label + '_std'])
                else:
                    experiment_scores_std_list.append(0.0)
            experiment_scores = (experiment_scores_list, experiment_scores_std_list)
            metric_scores[experiment_label] = experiment_scores
        scores[metric_label] = metric_scores
    data.append(scores)
    return data