import numpy as np
from collections import Counter
from .model import get_cluster_acc

def convert_from_dict(predicted_labels, d):
    for i in d.keys():
        predicted_labels[predicted_labels==i] = 10 + i

    for i in d.keys():
        predicted_labels[predicted_labels==(10 + i)] = d[i]

    return predicted_labels

def convert_predictions(true_labels, predicted_labels):
    counters = []
    convertion_dict = {}
    for i in np.unique(predicted_labels).tolist():
        new_counter = Counter(true_labels[predicted_labels==i])
        counters.append(new_counter)
        right_label = new_counter.most_common(1)[0][0]
        convertion_dict[i] = right_label
    new_predicted_labels = convert_from_dict(predicted_labels, convertion_dict)
    return new_predicted_labels

## TODO: do based on logits, need to permute them according to convert_dict
def combine_predictions(models, mode='val'):
    individual_accs, predictions = [], []
    for model in models:
        d = model.full_eval(mode=mode)
        predicted_labels = d['predicted_labels']
        true_labels = d['true_labels']
        #logits = d['logits']
        individual_accs.append(get_cluster_acc(true_labels, predicted_labels))
        predicted_labels = convert_predictions(true_labels, predicted_labels)
        predictions.append(predicted_labels)
    counters = [Counter([pred[i] for pred in predictions]) for i in range(true_labels.shape[0])]
    final_predictions = np.array([c.most_common(1)[0][0] for c in counters])
    return individual_accs, predictions, true_labels, final_predictions