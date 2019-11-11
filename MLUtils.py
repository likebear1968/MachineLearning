#!/usr/bin/env python
# coding: utf-8

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def create_pipelines(estimators, scaler=None, reductor=None):
    '''
    パイプラインを構築する。
    estimators：モデルリスト
    scaler：標準化オブジェクト
    reductor：次元削減オブジェクト
    戻り値：構築したパイプライン
    '''
    pipelines ={}
    for est in estimators:
        steps = []
        if scaler is not None: steps.append(('scl', scaler))
        if reductor is not None: steps.append(('rdt', reductor))
        k, v = est
        steps.append(('est', v))
        pipelines[k] = Pipeline(steps)
    return pipelines

def get_params(pipeline, upd={}):
    '''
    グリッドサーチ用のパラメータ群を取得する。
    pipeline：対象パイプライン
    upd：更新する個別パラメータ　※初期値は空の辞書
    戻り値：グリッドサーチ用のパラメータ群
    '''
    params = {k: [v] for k, v in pipeline.get_params().items() if '__' in k}
    for k, v in upd.items():
        if k in params:
            params[k] = v
    return params

def scaler(stype=9):
    '''
    スケーラを取得する。
    stype：スケーラのタイプ[0：StandardScaler、1：MinMaxScaler]　※初期値は9[None]
    戻り値：スケーラオブジェクト
    '''
    if stype == 0: return StandardScaler()
    elif stype == 1: return MinMaxScaler()
    return None

def reductor(rtype=9, k=10):
    '''
    次元削減オブジェクトを取得する。
    rtype：スケーラのタイプ[0：PCA、1：RFE]　※初期値は9[None]
    戻り値：次元削減オブジェクト
    '''
    if rtype == 0: return PCA(n_components=k)
    elif rtype == 1: return RFE(estimator=RandomForestClassifier, n_features_to_select=k)
    return None