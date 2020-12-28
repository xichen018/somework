#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras.backend as K

def precision(y_true, y_pred, threshold=.5):
    y_true_bool = K.equal(y_true, K.ones_like(y_true))
    y_pred_bool = y_pred >= threshold
    stacked_bool = K.stack([y_true_bool, y_pred_bool])
    correct_bool = K.all(stacked_bool, axis=0)
    correct_float = K.cast(correct_bool, 'float32')
    correct_num = K.sum(correct_float)

    y_pred_float = K.cast(y_pred_bool, 'float32')
    precision_total = K.sum(y_pred_float)
    result = correct_num / precision_total
    return result


def recall(y_true, y_pred, threshold=.5):
    y_true_bool = K.equal(y_true, K.ones_like(y_true))
    y_pred_bool = y_pred >= threshold
    stacked_bool = K.stack([y_true_bool, y_pred_bool])
    correct_bool = K.all(stacked_bool, axis=0)
    correct_float = K.cast(correct_bool, 'float32')
    correct_num = K.sum(correct_float)

    recall_total = K.sum(y_true)
    result = correct_num / recall_total
    return result


def exact_match_acc(y_true, y_pred, threshold=.5):
    y_true_bool = K.equal(y_true, K.ones_like(y_true))
    y_pred_bool = y_pred >= threshold
    matrix_bool = K.equal(y_true_bool, y_pred_bool)
    matrix_float = K.cast(matrix_bool, 'float32')
    prod = K.prod(matrix_float, axis=1)
    correct = K.sum(prod)

    total = K.shape(y_pred)[0]
    total = K.cast(total, 'float32')
    result = correct / total
    return result
