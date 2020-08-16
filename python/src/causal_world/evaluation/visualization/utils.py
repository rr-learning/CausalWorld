
def data_array_from_experiments(experiments):
    """
    Returns a structured list that can be processed by the plotters
    first element is a list of the protocol_names
    second element is a list of the experiment_labels
    third element is a list of the metric_labels
    fourth element is the list containing the scores:
        each element is a tuple of (metric_label, metric_scores)
            metric_scores is a tuple of (experiment_label, experiment_list)
                experiment_list is a list of scores for each protocol
    """

    experiment_labels = list(experiments.keys())
    protocol_labels = list(experiments[list(
        experiments.keys())[0]].keys())
    metric_labels = []
    for label in experiments[experiment_labels[0]][protocol_labels[0]].keys():
        if 'mean' in label:
            metric_labels.append(label)

    data = list()
    data.append(protocol_labels)
    data.append(experiment_labels)
    data.append(metric_labels)
    scores = list()
    for metric_label in metric_labels:
        metric_scores = list()
        for experiment_label in experiment_labels:
            experiment_list = list()
            for evaluation_protocol in protocol_labels:
                experiment_list.append(
                    experiments[experiment_label][evaluation_protocol][metric_label])
            experiment_scores = (experiment_label, experiment_list)
            metric_scores.append(experiment_scores)
        metric_data = (metric_label, metric_scores)
        scores.append(metric_data)
    data.append(scores)
    return data
