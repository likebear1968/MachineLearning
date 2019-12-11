import numpy as np
from abc import ABCMeta, abstractmethod
import itertools
from enum import Enum

class Factory():
    class MTYPE(Enum): R2=1; MSE=2; MAE=3; ACC=4; PCS=5; RCL=6; F1=7; MTR=8;
    @staticmethod
    def create(mtype, lbl=[0,1]):
        if mtype == Factory.MTYPE.R2:  return R2()
        if mtype == Factory.MTYPE.MSE: return MSE()
        if mtype == Factory.MTYPE.MAE: return MAE()
        if mtype == Factory.MTYPE.ACC: return Accuracy()
        if mtype == Factory.MTYPE.PCS: return Precision()
        if mtype == Factory.MTYPE.RCL: return Recall()
        if mtype == Factory.MTYPE.F1:  return F1()
        if mtype == Factory.MTYPE.MTR: return Confusion_matrix(lbl)
        return None

class Metrics(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, obs, prd):
        '''
        評価値を取得する。
        obs：実測値
        prd：予測値
        戻り値：評価値
        '''
        pass

class R2(Metrics):
    def evaluate(self, obs, prd):
        return 1 - np.sum((obs - prd) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

class MSE(Metrics):
    def evaluate(self, obs, prd):
        return np.average((obs - prd) ** 2, axis=0)

class MAE(Metrics):
    def evaluate(self, obs, prd):
        return np.average(np.abs(prd - obs), axis=0)

class Accuracy(Metrics):
    def evaluate(self, obs, prd):
        return np.average(obs == prd)

class Confusion_matrix(Metrics):
    def __init__(self, lbl=[0,1]):
        self.lbl = lbl

    def evaluate(self, obs, prd):
        if isinstance(obs, list): obs = np.array(obs)
        if isinstance(prd, list): prd = np.array(prd)
        matrix = []
        for a, b in itertools.product(self.lbl, repeat=2):
            matrix.append(len(np.where((obs == a) & (prd == b))[0]))
        return matrix

class Precision(Metrics):
    def evaluate(self, obs, prd):
        tn, fp, fn, tp = Confusion_matrix().evaluate(obs, prd)
        if tp + fp == 0: return tp
        return tp / (tp + fp)

class Recall(Metrics):
    def evaluate(self, obs, prd):
        tn, fp, fn, tp = Confusion_matrix().evaluate(obs, prd)
        if tp + fn == 0: return tp
        return tp / (tp + fn)

class F1(Metrics):
    def evaluate(self, obs, prd):
        tn, fp, fn, tp = Confusion_matrix().evaluate(obs, prd)
        if tp + fp + fn == 0: return (2 * tp) / 2
        return (2 * tp) / (2 * tp + fp + fn)
