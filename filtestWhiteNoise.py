# -*- coding: utf-8 -*-
"""
バタワースフィルタ設計テストとホワイトノイズでの評価
Created on Thu Dec 26 13:49 2025

@author: Teruki Toya, University of Yamanashi
"""

# %%
## モジュールのインポート ---------------------------------------------------
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

## 関数群 -----------------------------------------------------------------
# (ローパスフィルタ)

# (テーパー)
def taper_(x, N_tp):
    """
    < 入力 >
        x: テーパーをかけたい時間波形（ndarray）
        N_tp: テーパー部（上昇部）のサンプル数
    < 出力 >
        y: テーパー処理された時間波形（ndarray）
    """
    ctaper = np.sin(np.linspace(0, np.pi/2, N_tp))  # 上昇部のテーパー係数
    ctaper = np.concatenate([
                ctaper,
                np.ones(len(x) - 2*N_tp),
                np.flipud(ctaper)
             ])  # 下降部は上昇部の時間反転数列
    y = x * ctaper
    return y

#（ハニング窓）
def winHann(winSize):
    """
    < 入力 >
        winSize: 窓長（サンプル数）
    < 出力 >
        w: ハニング窓の係数列
    """
    Nw = np.linspace(0, 1, winSize)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * Nw)
    return w

# （シフトしながら短時間フレームに分割）
def shiftFrmDiv(x, winSize, shiftSize=-9999):
    """
    < 入力 >
        x: 時間波形（ndarray）
        winSize: 窓長（サンプル数）
        [shiftSize: フレームシフト長（既定: 窓長の半分）]
    < 出力 >
        y_frm: フレームごとに分割された時間波形 [フレーム数 × 窓長]
    """
    if shiftSize < 0:
        shiftSize = winSize // 2
    
    # 総フレーム数
    N_frm = int((len(x) - (winSize - shiftSize)) // shiftSize)
    
    # フレームごとに分割
    y_frm = np.zeros((N_frm, winSize))
    for i_frm in range(N_frm):
        offset = shiftSize * i_frm  # フレームをシフトしながら
        y_tmp = x[offset : offset + winSize]
        y_frm[i_frm, :] = y_tmp
    
    return y_frm
        

## ホワイトノイズの生成 -------------------------------------------------
fs = 48000  # サンプリング周波数 [Hz]
T = 2.0     # 信号時間長 [s]
xt = np.random.randn(round(fs*T))       # ガウス性白色雑音
xt = 0.75 * (xt / np.max(np.abs(xt)))
xt = taper_(xt, round(fs/20))           # テーパー処理

## ホワイトノイズへのフィルタ適用 ----------------------------------------
f_lpCut = 2000  # LPFのカットオフ周波数 [Hz]
w_lpCut = 2*np.pi * f_lpCut
f_hpCut = 1000  # HPFのカットオフ周波数 [Hz]
w_hpCut = 2*np.pi * f_hpCut

bl, al = signal.butter(4, w_lpCut, 'low')  # バタワース型LPFのフィルタ係数
w, h = signal.freqs(bl, al)
plt.figure(w, 20 * np.log10(abs(h)))
plt.title('LPF frequency response')
plt.xlabel('Norm. angular frequency')
plt.ylabel('Gain [dB]')
plt.axvline(w_lpCut, color='green')
plt.show()
# %%
