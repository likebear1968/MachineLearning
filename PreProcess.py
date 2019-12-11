#!/usr/bin/env python
# coding: utf-8

import numpy as np

def to_dummy(data, idx, cord=None, has_nan=False):
    '''
    区分値コードをダミー変数に変換する。
    data：データ行列
    idx：対象列ID
    cord：区分値リスト
    戻り値：ダミー変数化した行列
    '''
    ndat = np.array(data).reshape(-1, np.shape(data)[-1])[:, idx]
    dmmy = np.array([] * ndat.shape[0]).reshape(ndat.shape[0], -1)
    crds = []
    if has_nan:
        mask = ndat == ndat
    else:
        mask = ndat != 'nan'
    for i in range(ndat.shape[-1]):
        if cord is None:
            c = np.unique(ndat[:, i][mask[:, i]])
        else:
            c = np.array(cord)
        col = np.zeros((ndat.shape[0], len(c)))
        for j, v in enumerate(c):
            col[:, j][ndat[:, i] == v] = 1
        crds.append(c.tolist())
        dmmy = np.append(dmmy, col, axis=1)
        
    return crds, dmmy

def scaling(data, idx):
    '''
    標準化(平均0、分散1)した列データを取得する。
    data：データ行列
    idx：対象列IDリスト
    戻り値：標準化した列データ
    '''
    ndat = data[:, idx].astype(float)
    ndat -= np.nanmean(ndat, axis=0)
    ndat /= np.nanstd(ndat, axis=0)
    return ndat

def imputing(data, idx, strategy=0):
    '''
    欠損値を補完する。
    data：データ行列
    idx：対象列IDリスト
    strategy：補完する値の種類[0：平均値、1：中央値]※初期値は0
    戻り値：欠損値を補完した行列
    '''
    ndat = data[:, idx].astype(float)
    vals = np.zeros(data.shape[1])
    if strategy == 0:
        vals = np.nanmean(ndat, axis=0)   # 平均値
    elif strategy == 1:
        vals = np.nanmedian(ndat, axis=0) # 中央値
    for i in range(len(idx)):
        ndat[:, i][np.isnan(ndat[:, i])] = vals[i]
    return ndat

def to_float(data):
    for i in range(np.shape(data)[-1]):
        try:
            data[:, i].astype(float)
        except ValueError:
            data[:, i] = 0
    return data.astype(float)

def describe(data, idx=None):
    '''
    統計情報を取得する。
    data：データ行列
    idx：対象列IDリスト
    戻り値：nan以外のデータ件数
    　　　　nanのデータ件数
        　　ユニーク件数
    　　　　最小値
        　　最大値
          　平均値
        　　中央値
          　標準偏差
        　　下限値(25%タイル - IQR)
          　上限値(75%タイル + IQR)
    '''
    row = 10
    if idx is None:
        if np.ndim(data) == 1:
            idx = list(range(len(data)))
        else:
            idx = list(range(np.shape(data)[-1]))
    ndat = np.array(data).reshape(-1, np.shape(data)[-1])[:, idx].astype(str)
    stat = np.zeros((row, len(idx)))
    mask = ndat == 'nan'
    nans = np.count_nonzero(mask, axis=0)
    stat[0][...] = ndat.shape[0] - nans                      # nan以外のデータ件数
    stat[1][...] = nans                                      # nanのデータ件数
    stat[2][...] = [len(np.unique(ndat[:, i][~mask[:, i]])) for i in range(len(idx))] # ユニーク件数
    ndat = to_float(ndat)
    stat[3][...] = np.nanmin(ndat, axis=0)                   # 最小値
    stat[4][...] = np.nanmax(ndat, axis=0)                   # 最大値
    stat[5][...] = np.nanmean(ndat, axis=0)                  # 平均値
    stat[6][...] = np.nanmedian(ndat, axis=0)                # 中央値
    stat[7][...] = np.nanstd(ndat, axis=0)                   # 標準偏差
    q25, q75 = np.nanpercentile(ndat, q=[25, 75], axis=0)
    iqr = (q75 - q25) * 1.5
    stat[8][...] = q25 - iqr                                 # 下限値
    stat[9][...] = q75 + iqr                                 # 上限値
    return stat

def unique(data, idx):
    '''
    項目件数を取得する。
    data：データ行列
    idx：対象列IDリスト
    戻り値：項目件数
    '''
    uq = []
    for i in idx:
        items, counts = np.unique(data[:, i], return_counts=True)
        uq.append([[v, c] for v, c in zip(items, counts)])
    return uq