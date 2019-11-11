#!/usr/bin/env python
# coding: utf-8

import numpy as np

def to_dummy(data, idx, vals, has_nan=False):
    '''
    区分値コードをダミー変数に変換する。
    data：データ行列
    idx：対象列ID
    vals：区分値リスト
    戻り値：ダミー変数化した行列
    '''
    col = len(vals)
    if has_nan: col += 1
    dummies = np.zeros((data.shape[0], col), dtype=int)
    column = data[:, idx].astype(str)
    for i, v in enumerate(column):
        if v == 'nan':
            dummies[i, -1] = 1
        else:
            #if v.replace('.', '').replace('-', '').isdigit(): v = str(int(float(v)))
            if v not in vals:
                print(f'{v}は区分値リストに存在しません。')
            else:
                dummies[i, vals.index(v)] = 1
    return dummies

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

def describe(data, idx):
    '''
    統計情報を取得する。
    data：データ行列
    idx：対象列IDリスト
    戻り値：nan以外のデータ件数
    　　　　nanのデータ件数
    　　　　最小値
        　　最大値
          　平均値
        　　中央値
          　標準偏差
        　　下限値(25%タイル - IQR)
          　上限値(75%タイル + IQR)
    '''
    ndat = data[:, idx].astype(float)
    stat = np.zeros((9, len(idx)))
    nans = np.count_nonzero(np.isnan(ndat), axis=0)
    stat[0][...] = ndat.shape[0] - nans                      # nan以外のデータ件数
    stat[1][...] = nans                                      # nanのデータ件数
    stat[2][...] = np.nanmin(ndat, axis=0)                   # 最小値
    stat[3][...] = np.nanmax(ndat, axis=0)                   # 最大値
    stat[4][...] = np.nanmean(ndat, axis=0)                  # 平均値
    stat[5][...] = np.nanmedian(ndat, axis=0)                # 中央値
    stat[6][...] = np.nanstd(ndat, axis=0)                   # 標準偏差
    q25, q75 = np.nanpercentile(ndat, q=[25, 75], axis=0)
    iqr = (q75 - q25) * 1.5
    stat[7][...] = q25 - iqr                                 # 下限値
    stat[8][...] = q75 + iqr                                 # 上限値
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