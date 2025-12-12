# -*- coding: utf-8 -*-
"""
感性情報工学及び演習 - 聴覚実験用プログラム
Created on Thu Dec 11 14:36 2025
Last updated on Thu Dec 11 14:36 2025

@author: Teruki Toya, University of Yamanashi
"""

# モジュールのインポート
import flet as ft
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal

## 固定パラメータの設定 -----------------------------------------------
#sd.default.device = [1, 4]
# フィルタカットオフ周波数 [Hz]
fH2 = 15000
fH3 = 8000
fH4 = 3400
fL5 = 300
fH5 = 3400
fL6 = 1000
fH6 = 3400
# 刺激         fL      fH
#   1 (CD)    ----    20000
#   2 (FM)    ----    15000
#   3 (AM)    ----     8000
#   4 (s1)    ----     3400
#   5 (TEL)    300     3400
#   6 (s2)    1000     3400

# フィルタの通過/阻止域ゲイン損失 [dB]
g_pass = 3
g_stop = 40

## 関数群 -----------------------------------------------------------------
# (ローパスフィルタ)
def lowpass(x, fs, f_pass, g_pass, g_stop):
    f_stop = f_pass + 300           # 阻止域端周波数
    w_pass = f_pass / (fs/2)        # ナイキスト周波数で通過域端周波数を正規化
    w_stop = f_stop / (fs/2)        # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(w_pass, w_stop, g_pass, g_stop)  # フィルタ次数とバタワース正規化周波数を計算
    b, a = signal.butter(N, Wn, 'lowpass')                  # フィルタ伝達関数の分子と分母を計算
    
    

# Flet の処理 ---------------------
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
    trial_disp = ft.Ref[ft.Text]()        # 試行数表示部を定義
    ans_radio = ft.Ref[ft.RadioGroup]()   # 回答部ラジオボタンを定義

    # OKボタン押下時の動作を記述 -----------------------------------
    def button_clicked(e):
        # 原音を読み込む
        x, fs = sf.read('02AuraLee3.mp3')
        # 音を出す
        sd.play(x, fs)
        
        page.update()               # ページを更新

    # Flet コントロールの追加とページへの反映 -------------------------------
    page.add(
        ft.Row(
            controls=[
                ft.TextField(                                  # ID入力テキストボックス
                    ref=ID_txtbox,
                    label="実験参加者ID",
                ),
                ft.DropdownM2(                                 # 予備/本実験選択窓
                    ref=exp_drpdn,
                    width=150,
                    options=[
                        ft.dropdownm2.Option("予備実験"),
                        ft.dropdownm2.Option("本実験"),
                    ],
                ),
                ft.Text("　"),
                ft.Text(
                    "0/0",
                    ref=trial_disp,
                    size=25,
                )
            ]
        ),
        ft.Text(""),                                           # 空行
        ft.Text("先行刺激(A)と後続刺激(B)を比べて音質は……"),      # 回答部テキスト
        ft.RadioGroup(                                         # 回答部ラジオボタン
           ref=ans_radio,
           content=ft.Row(
               [
                   ft.Radio(value="2", label="Aの方が良い"),
                   ft.Radio(value="1", label="ややAの方が良い"),
                   ft.Radio(value="0", label="どちらともいえない"),
                   ft.Radio(value="-1", label="ややBの方が良い"),
                   ft.Radio(value="-2", label="Bの方が良い"),
               ]
           )
        ),

        
        # OKボタン
        ft.ElevatedButton("OK", on_click = button_clicked),
        
    )

ft.app(target=main)
