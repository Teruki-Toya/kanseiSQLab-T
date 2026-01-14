# -*- coding: utf-8 -*-
"""
スぺクトルの図示
Created on Thu Jan. 14 19:57 2026

@author: Teruki Toya, University of Yamanashi
"""

# %%
## 解析したい刺激ファイル名の宣言
# *******************************
sndName = 'stim2'
# *******************************

## モジュールのインポート ---------------------------------------------------
import numpy as np
import soundfile as sf
from scipy import signal
import scipy.fftpack as spfft
import matplotlib.pyplot as plt

## スペクトル解析 -----------------------------------------------------------
x, fs = sf.read(sndName+'.wav')    # wav読込
x = x[:, 0]                 # ステレオ化
xlen = len(x)               # 音刺激のサンプル長

Xf = spfft.fft(x)           # 高速フーリエ変換
Xf_amp = np.abs(Xf)         # 振幅スペクトル（複素数の絶対値）

freq = np.linspace(0, xlen - 1, xlen) * fs / xlen  # 周波数軸
Xf_dB = 20 * np.log10(Xf_amp / np.max(Xf_amp))     # 最大パワーを基準とした相対レベル [dB]

freq = freq[:xlen//2]       # 正の周波数のみ
Xf_dB = Xf_dB[:xlen//2]

## 図示
plt.plot(freq, Xf_dB, lw = 0.75)
plt.xlabel('Frequency [Hz]', fontsize=15)
plt.ylabel('Relative level [dB]', fontsize=15)
plt.xscale('log')
plt.xlim([100, 20000])
plt.xticks(
        [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        labels=['125', '250', '500', '1000', '2000', '4000', '8000', '16000'],
        minor=False
        )
plt.ylim([-60, 0])
plt.tick_params(axis='both', which='major', labelsize=13)

plt.savefig('sp_'+sndName+'.png', format='png', bbox_inches='tight')
plt.show()
# %%
