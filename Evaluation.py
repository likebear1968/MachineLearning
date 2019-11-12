#!/usr/bin/env python
# coding: utf-8

import numpy as np
import itertools

def R2(obs, prd):
    '''
    R2(決定係数)を取得する。
    obs：実測値
    prd：予測値
    戻り値：誤差
    '''
    return 1 - np.sum((obs - prd) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def MSE(obs, prd):
    '''
    平均二乗誤差を取得する。
    obs：実測値
    prd：予測値
    戻り値：誤差
    '''
    return np.average((obs - prd) ** 2, axis=0)

def MAE(obs, prd):
    '''
    平均絶対誤差を取得する。
    obs：実測値
    prd：予測値
    戻り値：誤差
    '''
    return np.average(np.abs(prd - obs), axis=0)

def accuracy(obs, prd):
    '''
    正解率を取得する。
    obs：実測値
    prd：予測値
    戻り値：正解率
    '''
    if isinstance(obs, list): obs = np.array(obs)
    if isinstance(prd, list): prd = np.array(prd)
    return np.average(obs == prd)

def precision(obs, prd):
    '''
    精度を取得する。
    obs：実測値
    prd：予測値
    戻り値：精度
    '''
    tn, fp, fn, tp = confusion_matrix(obs, prd)
    if tp + fp == 0: return tp
    return tp / (tp + fp)

def recall(obs, prd):
    '''
    検出率を取得する。
    obs：実測値
    prd：予測値
    戻り値：検出率
    '''
    tn, fp, fn, tp = confusion_matrix(obs, prd)
    if tp + fn == 0: return tp
    return tp / (tp + fn)

def f1(obs, prd):
    '''
    F値を取得する。
    obs：実測値
    prd：予測値
    戻り値：F値
    '''
    tn, fp, fn, tp = confusion_matrix(obs, prd)
    if tp + fp + fn == 0: return (2 * tp) / 2
    return (2 * tp) / (2 * tp + fp + fn)

def confusion_matrix(obs, prd, lbl=[0,1]):
    '''
    混同行列を取得する。
    obs：実測値
    prd：予測値
    lbl：多クラスの場合に使用するラベルリスト
    戻り値：混同行列[tn, fp, fn, tp]
    '''
    if isinstance(obs, list): obs = np.array(obs)
    if isinstance(prd, list): prd = np.array(prd)
    matrix = []
    for a, b in itertools.product(lbl, repeat=2):
        matrix.append(len(np.where((obs == a) & (prd == b))[0]))
    return matrix
