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
import scipy.fftpack as spfft
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

# （長時間平均スペクトル）
def LTAS(x, winSize):
    """
    < 入力 >
        x: 時間波形（ndarray）
        winSize: 窓長（サンプル数）
    < 出力 >
        Yf_dB_mean: 長時間平均（振幅）スペクトル
    """
    y_frm = shiftFrmDiv(x, winSize)  # 窓ごとに分割（半窓長重畳）
    Yf_frm = spfft.fft(y_frm, winSize, 1)  # 列方向にFFT
    Yf_dB_frm = 20 * np.log10(np.abs(Yf_frm))  # 振幅のdB表現
    Yf_dB_mean = np.mean(Yf_dB_frm, 0)  # フレーム方向に平均
    Yf_dB_mean = Yf_dB_mean[:winSize//2]  # 正の周波数を抽出

    return Yf_dB_mean

## フィルタ設計 -----------------------------------------
#（LPF）
fs = 48000  # サンプリング周波数 [Hz]
f_lpCut = 3400  # LPFのカットオフ周波数 [Hz]
w_lpCut = f_lpCut / (fs/2)

bl, al = signal.butter(4, w_lpCut, 'low')  # バタワース型LPFのフィルタ係数
w, H = signal.freqz(bl, al)
plt.plot((fs/2)*w/np.pi, 20 * np.log10(np.abs(H)))
plt.title('LPF frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.axvline(f_lpCut, color='green')
plt.xlim(0, 20000)
plt.ylim(-65, 5)
plt.show()

#（HPF）
f_hpCut = 1000  # HPFのカットオフ周波数 [Hz]
w_hpCut = f_hpCut / (fs/2)

bh, ah = signal.butter(4, w_hpCut, 'high')  # バタワース型HPFのフィルタ係数
w, H = signal.freqz(bh, ah)
plt.plot((fs/2)*w/np.pi, 20 * np.log10(np.abs(H)))
plt.title('HPF frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.axvline(f_hpCut, color='green')
plt.xlim(0, 20000)
plt.ylim(-65, 5)
plt.show()
        
## ホワイトノイズの生成 -------------------------------------------------
T = 2.0     # 信号時間長 [s]
xt = np.random.randn(round(fs*T))       # ガウス性白色雑音
xt = 0.75 * (xt / np.max(np.abs(xt)))
xt = taper_(xt, round(fs/20))           # テーパー処理

winSize = 1024  # 分析窓長
Xf_dB = LTAS(xt, winSize)      # 長時間平均スペクトル
Xf_dB = Xf_dB - np.max(Xf_dB)  # 最大値で正規化
freq = np.linspace(0, winSize - 1, winSize) * fs / winSize
freq = freq[:winSize//2]       # 周波数軸

## ホワイトノイズへのフィルタ適用 ----------------------------------------
yt = signal.lfilter(bl, al, xt)  # LPFのフィルタリング
yt = signal.lfilter(bh, ah, yt)  # HPFのフィルタリング
Yf_dB = LTAS(yt, winSize)      # 長時間平均スペクトル
Yf_dB = Yf_dB - np.max(Yf_dB)  # 最大値で正規化

plt.plot(freq, Xf_dB, color = "blue", label = "Original")
plt.plot(freq, Yf_dB, color = "red", label = "Filtered")
plt.title('White noise spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Relative level [dB]')
plt.xlim(0, 20000)
plt.ylim(-65, 5)
plt.legend
plt.show()

# %%
