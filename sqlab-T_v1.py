# -*- coding: utf-8 -*-
"""
感性情報工学及び演習 - 聴覚実験用プログラム
Created on Thu Dec 11 14:36 2025
Last updated on Thu Dec 11 14:36 2025

@author: Teruki Toya, University of Yamanashi
"""

# %%
## モジュールのインポート ---------------------------------------------------
import flet as ft
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import pandas as pd
import time
import datetime

# %%

## 関数群 -----------------------------------------------------------------
# (ローパスフィルタ)
def lowpass(x, fs, f_cut, ord):
    """
    < 入力 >
        x: 時間波形（ndarray: サンプル × 2ch(LとR)）
        fs: サンプリング周波数 [Hz]
        f_cut: カットオフ周波数 [Hz]
        ord: フィルタ次数
    < 出力 >
        y: フィルタリングされた時間波形（xと同次元）
    """
    w_cut = f_cut / (fs/2)        # ナイキスト周波数で通過域端周波数を正規化
    b, a = signal.butter(ord, w_cut, 'low')  # バタワース型LPFのフィルタ係数                 # フィルタ伝達関数の分子と分母を計算
    y = signal.lfilter(b, a, x, axis=0)      # 入力 x にフィルタを適用
    return y

# (ハイパスフィルタ)
def highpass(x, fs, f_cut, ord):
    """
    < 入力 >
        x: 時間波形（ndarray: サンプル × 2ch(LとR)）
        fs: サンプリング周波数 [Hz]
        f_cut: カットオフ周波数 [Hz]
        ord: フィルタ次数
    < 出力 >
        y: フィルタリングされた時間波形（xと同次元）
    """
    w_cut = f_cut / (fs/2)        # ナイキスト周波数で通過域端周波数を正規化
    b, a = signal.butter(ord, w_cut, 'high')  # バタワース型HPFのフィルタ係数                 # フィルタ伝達関数の分子と分母を計算
    y = signal.lfilter(b, a, x, axis=0)              # 入力 x にフィルタを適用
    return y

## 固定パラメータ・変数の設定 -----------------------------------------------
#sd.default.device = [1, 4]

# (フィルタカットオフ周波数 [Hz])
fH2 = 15000
fH3 = 8000
fH4 = 3400
fL5 = 300
fH5 = 3400
fL6 = 1000
fH6 = 3400
# (フィルタ次数)
ordH2 = 8
ordH3 = 8
ordH4 = 4
ordL5 = 4
ordH5 = 4
ordL6 = 4
ordH6 = 4
# 刺激         fL (ord.)      fH (ord.)
#   1 (CD)    ----             20000
#   2 (FM)    ----             15000 (8)
#   3 (AM)    ----              8000 (8)
#   4 (s1)    ----              3400 (4)
#   5 (TEL)    300 (4)          3400 (4)
#   6 (s2)    1000 (4)          3400 (4)

## 提示刺激の設定 -----------------------------------------------
#（刺激の生成）
x1, fs = sf.read('02AuraLee3.mp3')         # 原音（= 刺激1）
x1 = x1[:, 0]

x2 = lowpass(x1, fs, fH2, ordH2)  # 刺激2
x3 = lowpass(x1, fs, fH3, ordH3)  # 刺激3
x4 = lowpass(x1, fs, fH4, ordH4)  # 刺激4

x5 = lowpass(x1, fs, fH5, ordH5)
x5 = highpass(x5, fs, fL5, ordL5)  # 刺激5

x6 = lowpass(x1, fs, fH6, ordH6)
x6 = highpass(x6, fs, fL6, ordL6)  # 刺激6

#（RMS調整）
x1 = 0.75 * (x1 / np.max(np.abs(x1)))       # 基準振幅を決めておく
RMSref = np.sqrt(np.mean(x1 ** 2))          # 基準RMS

RMS_x2 = np.sqrt(np.mean(x2 ** 2))          # RMS
RMS_x3 = np.sqrt(np.mean(x3 ** 2))
RMS_x4 = np.sqrt(np.mean(x4 ** 2))
RMS_x5 = np.sqrt(np.mean(x5 ** 2))
RMS_x6 = np.sqrt(np.mean(x6 ** 2))
x2 = (RMSref/RMS_x2) * x2                   # RMS調整
x3 = (RMSref/RMS_x3) * x3
x4 = (RMSref/RMS_x4) * x4
x5 = (RMSref/RMS_x5) * x5
x6 = (RMSref/RMS_x6) * x6

#（結合）
x = np.stack([x1, x2, x3, x4, x5, x6], axis=1)  # 6-ch 信号

# %%
# Flet の処理 --------------------------------------------------
def main(page):
    
    page.title = "SQLab-T Version 1.0"  # タイトル
    # page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.window_width = 500  # 幅
    page.window_height = 350  # 高さ
    page.window_top = 100  # 位置(TOP)
    page.window_left = 100  # 位置(LEFT)
    page.window_always_on_top = True  # ウィンドウを最前面に固定
    #page.window_center()  # ウィンドウをデスクトップの中心に移動
    
    ID_txtbox = ft.Ref[ft.TextField]()    # ID入力テキストボックスを定義
    exp_drpdn = ft.Ref[ft.DropdownM2]()   # 予備/本実験選択窓を定義
    init_button = ft.Ref[ft.Button]()     # 初期化ボタンを定義
    trial_disp = ft.Ref[ft.Text]()        # 試行数表示部を定義
    ans_rg = ft.Ref[ft.RadioGroup]()   # 回答部ラジオボタンを定義
    OK_button = ft.Ref[ft.Button]()       # OKボタンを定義
    
    # Init ボタン押下時の動作を記述 -----------------------------------
    def buttonInit_clicked(e):
        ID_txtbox.current.disabled = True
        exp_drpdn.current.disabled = True
        init_button.current.disabled = True
        OK_button.current.disabled = False
        if exp_drpdn.current.value == "本実験":
            N_stim = 6  # 本実験ならば刺激種類 6
        else:
            N_stim = 3  # 予備実験ならば刺激種類 3
        Ns_sr = pd.Series(N_stim)  # pandasシリーズ化
        
        # 回答結果保存用データフレームの生成
        res_df = pd.DataFrame(columns=['Participant', 'First Stimulus', 'Second Stimulus', 'Trial', 'Result'])
        now = datetime.datetime.now()
        currentDateTime = now.strftime("%m%d%H%M")
        resCsvFileName = 'SQResult-'+currentDateTime+'.csv'
        res_df.to_csv(resCsvFileName, index = False)
        csvFN_sr = pd.Series(resCsvFileName)
        
        # 刺激ペア生成用数列
        K = np.random.permutation(np.arange(N_stim ** 2))
        K = K[K % (N_stim + 1) > 0]
        K_sr = pd.Series(K)      # pandasシリーズ化
        # 実験試行数カウンタ
        count = 0
        c_sr = pd.Series(count)  # pandasシリーズ化
        # KとN_stim、count、csvFNを結合
        set_df = pd.concat([K_sr, Ns_sr, c_sr, csvFN_sr], axis = 1)
        set_df.columns = ['Kk', 'Ns', 'Cnt', 'csvFN']
        # 設定用数列データとして保存
        set_df.to_csv('set.csv', index = False)
        
        page.update()               # ページを更新

    # OKボタン押下時の動作を記述 -----------------------------------
    def buttonOK_clicked(e):
        ans_rg.current.disabled = True  # 回答不能にする
        page.update()               # ページを更新
        
        # 前試行までの K, N_stim, countとcsvFileNameを読込
        set_df = pd.read_csv('set.csv')
        K = set_df.Kk.to_numpy()
        N_stim = int(set_df.Ns.to_numpy()[0])
        count = int(set_df.Cnt.to_numpy()[0])
        csvFileName = set_df.csvFN[0]
        # 実験試行数の更新
        count += 1
        trial_disp.current.value = str(count) + "/" + str(len(K))

        # 刺激ペアを決定
        firstStim = K[count-1] // N_stim   # 【先再生】の刺激番号
        secondStim = K[count-1] % N_stim   # 【後再生】の刺激番号
        if exp_drpdn.current.value == "予備実験":
            firstStim = int(np.trunc(firstStim * 2.5))   # 刺激番号として0,2,5を指定
            secondStim = int(np.trunc(secondStim * 2.5))
        print(str(count) + "/" + str(len(K)) + ": 【刺激" + str(firstStim) + "】v.s.【刺激" + str(secondStim) + "】")
        
        page.update()               # ページを更新
        
        # 音を出す
        sd.play(x[:, firstStim], fs)  # 先行刺激（A）を流す
        time.sleep(8)
        sd.play(x[:, secondStim], fs) # 後続刺激（B）を流す
        
        # count の更新・反映
        set_df.Cnt = pd.Series(count)  # pandasシリーズ化
        set_df.to_csv('set.csv', index = False)
        
        ans_rg.current.disabled = False  # 回答可能にする
        page.update()                   # ページを更新
        
        # 実験結果の保存
        res_df = pd.read_csv(csvFileName)
        newData = pd.DataFrame({'Participant': [ID_txtbox.current.value],
                   'First Stimulus': [firstStim],
                   'Second Stimulus': [secondStim],
                   'Trial': [count],
                   'Result': [ans_rg.current.value],
                   })  # 回答結果
        res_df = pd.concat([res_df, newData])  # 現在の回答を追加
        res_df.to_csv(csvFileName, index = False)
        
        # 終了フラグ
        
        

    # Flet コントロールの追加とページへの反映 -------------------------------
    page.add(
        ft.Row(
            controls=[
                ft.TextField(                                  # ID入力テキストボックス
                    ref=ID_txtbox,
                    label="実験参加者ID"
                ),
                ft.DropdownM2(                                 # 予備/本実験選択窓
                    ref=exp_drpdn,
                    width=150,
                    options=[
                        ft.dropdownm2.Option("予備実験"),
                        ft.dropdownm2.Option("本実験"),
                    ]
                ),
                ft.Text("　"),
                ft.ElevatedButton(
                    "実験初期化",
                    ref=init_button,
                    on_click = buttonInit_clicked
                ),
                ft.Text(
                    "0/0",
                    ref=trial_disp,
                    size=25
                )
            ]
        ),
        ft.Text(""),                                        # 空行
        ft.Text("先行刺激(A)と後続刺激(B)を比べて音質は……"),      # 回答部テキスト
        ft.RadioGroup(                                      # 回答部ラジオボタン
           ref=ans_rg,
           content=ft.Row(
               [
                   ft.Radio(value="2", label="Aの方が良い"),
                   ft.Radio(value="1", label="ややAの方が良い"),
                   ft.Radio(value="0", label="どちらともいえない"),
                   ft.Radio(value="-1", label="ややBの方が良い"),
                   ft.Radio(value="-2", label="Bの方が良い"),
               ]
           ),
           disabled = True
        ),

        # OKボタン
        ft.ElevatedButton(
            "OK",
            ref = OK_button,
            on_click = buttonOK_clicked,
            disabled = True
        ),
        
    )

ft.app(target=main)
