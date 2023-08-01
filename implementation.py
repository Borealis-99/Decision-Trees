import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """
    total_correct = 0
    unique_feature_values = np.unique(x_inputs[:, feature_index]) # get the unique values of the feature

    for value in unique_feature_values: # for each value of the feature
        value_mask = (x_inputs[:, feature_index] == value) # get the samples that have that value
        value_labels = y_outputs[value_mask] # get the labels of those samples

        class_counts = [np.sum(value_labels == label) for label in classes] # count the number of each class
        max_class_count = max(class_counts) # get the most common class
        total_correct += max_class_count # add the number of samples with that class to the total

    return total_correct

    total_correct = 0
    
    for feature_value in set(x_inputs[:, feature_index]): # for each value of the feature
        class_counts = {c: 0 for c in classes} # count the number of each class
        for i in range(len(x_inputs)): # for each sample
            if x_inputs[i, feature_index] == feature_value: # if the sample has the feature value
                class_counts[y_outputs[i]] += 1 # increment the count for that class
        predicted_class = max(class_counts, key=class_counts.get) # predict the class with the most samples
        total_correct += sum(y_outputs[x_inputs[:, feature_index] == feature_value] == predicted_class) # count the number of samples with that class
        
    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """

    entropy = 0  # TODO: fix me
    n_samples = len(y_outputs)
    for label in classes:
        prob = np.sum(y_outputs == label) / n_samples
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """Calculate the information remainder after splitting the input-output set based on the
given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    n_samples = len(y_outputs)
    set_entropies = []  # TODO: fix me
    for value in np.unique(x_inputs[:, feature_index]):
        mask = (x_inputs[:, feature_index] == value)
        subset_x = x_inputs[mask]
        subset_y = y_outputs[mask]
        subset_entropy = set_entropy(subset_x, subset_y, classes)
        set_entropies.append((len(subset_y) / n_samples) * subset_entropy)

    # Calculate the remainder
    remainder = np.sum(set_entropies)  # TODO: fix me

    gain = overall_entropy - remainder  # TODO: fix me

    return gain
