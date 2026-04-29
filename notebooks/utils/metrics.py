import numpy as np


def class_errors(y, y_pred):
    classes = np.unique(y)
    mask = y[:, None] == classes[None, :]
    error_vector = mask & (y_pred[:, None] != classes)
    return np.sum(error_vector, axis=0) / np.sum(mask, axis=0)


def balanced_error(y, y_pred):
    return np.mean(class_errors(y, y_pred))


def worst_group_error(y, y_pred):
    return np.max(class_errors(y, y_pred))


def class_acc(y, y_pred):
    classes = np.unique(y)
    mask = y[:, None] == classes[None, :]
    correct_vector = mask & (y_pred[:, None] == classes)
    return np.sum(correct_vector, axis=0) / np.sum(mask, axis=0)


def balanced_acc(y, y_pred):
    return np.mean(class_acc(y, y_pred))


def worst_group_acc(y, y_pred):
    return np.min(class_acc(y, y_pred))
