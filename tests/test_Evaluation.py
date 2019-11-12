#!/usr/bin/env python
# coding: utf-8

import pytest
import Evaluation as ev

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import itertools

@pytest.mark.parametrize('y_pred', [[a, b] for a, b in itertools.product(range(2), repeat=2)])
def test_confusion_matrix(y_pred):
    y_true = [0, 1]
    expect = confusion_matrix(y_true, y_pred).flatten().tolist()
    assert ev.confusion_matrix(y_true, y_pred) == expect

@pytest.mark.parametrize('y_pred', [[a, b, c] for a, b, c in itertools.product(range(3), repeat=3)])
def test_confusion_matrix_multi(y_pred):
    y_true = [0, 1, 2]
    expect = confusion_matrix(y_true, y_pred, labels=y_true).flatten().tolist()
    assert ev.confusion_matrix(y_true, y_pred, lbl=y_true) == expect

@pytest.mark.parametrize('y_pred', [[a, b] for a, b in itertools.product(range(2), repeat=2)])
def test_accuracy(y_pred):
    y_true = [0, 1]
    expect = accuracy_score(y_true, y_pred)
    assert ev.accuracy(y_true, y_pred) == expect

@pytest.mark.parametrize('y_pred', [[a, b] for a, b in itertools.product(range(2), repeat=2)])
def test_precision(y_pred):
    y_true = [0, 1]
    expect = precision_score(y_true, y_pred)
    assert ev.precision(y_true, y_pred) == expect

@pytest.mark.parametrize('y_pred', [[a, b] for a, b in itertools.product(range(2), repeat=2)])
def test_recall(y_pred):
    y_true = [0, 1]
    expect = recall_score(y_true, y_pred)
    assert ev.recall(y_true, y_pred) == expect

@pytest.mark.parametrize('y_pred', [[a, b] for a, b in itertools.product(range(2), repeat=2)])
def test_f1(y_pred):
    y_true = [0, 1]
    expect = f1_score(y_true, y_pred)
    assert ev.f1(y_true, y_pred) == expect

if __name__ == '__main__':
    pytest.main(['-v', __file__])
