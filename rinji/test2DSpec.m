%% =========================================================
%  時空間音圧分布と時間平均2次元スペクトルの表示
%
%  ・時空間音圧分布：
%       解析対象信号の中央時刻に最も近い1フレームを表示
%
%  ・時空間2次元スペクトル：
%       全フレームのパワースペクトルを平均して表示
% ==========================================================

clear;
close all;
clc;

%% ---------------------------------------------------------
% 設定
% ----------------------------------------------------------

micSpacing = 0.042;           % マイクロホン間隔 [m]

% analysisStartTime = 0.0;     % 解析開始時刻 [s]
% analysisDuration  = 10.0;    % 解析対象の時間長 [s]

L_frm = 1024;        % 1フレームの時間長 [s]

displayDynamicRange = 80;    % スペクトル表示のダイナミックレンジ [dB]

%% ---------------------------------------------------------
% 多チャンネル信号の読み込み
% ----------------------------------------------------------

% WAVファイルの各列に各マイクロホン信号が格納されていると仮定
[x_array, fs] = audioread("move.wav");

if size(x_array, 2) < 20
    x_array = x_array';
end

% recordedData のサイズ：
%   [時間サンプル数 × チャンネル数]

[numTotalChannels, numTotalSamples] = size(x_array);

if numTotalChannels < 2
    error("2チャンネル以上の多チャンネル信号が必要です。");
end

x_analy = [x_array(6, :);  x_array(5, :);
           x_array(12, :); x_array(11, :)]; % 横方向

[m_ch, Ns] = size(x_analy);

% 線形・等間隔マイクロホンアレイを仮定
micPositions = (0 : m_ch - 1) .* micSpacing;

%% ---------------------------------------------------------
% 解析対象区間の切り出し
% ----------------------------------------------------------

analysisStartSample = 1;
numAnalysisSamples  = numTotalSamples;
analysisEndSample   = analysisStartSample ...
                    + numAnalysisSamples - 1;

if analysisStartSample < 1
    error("analysisStartTimeには0以上の値を指定してください。");
end

if analysisEndSample > numTotalSamples
    availableDuration = ...
        (numTotalSamples - analysisStartSample + 1) / fs;

    error([ ...
        "指定した解析区間が入力信号の長さを超えています。" ...
        newline ...
        "指定した解析時間長：%.3f s" ...
        newline ...
        "利用可能な解析時間長：%.3f s"], ...
        analysisDuration, availableDuration);
end

x_analy = x_analy(:, analysisStartSample:analysisEndSample);

%% ---------------------------------------------------------
% フレーム条件の設定
% ----------------------------------------------------------

hopLength     = round(L_frm/2);

if L_frm < 2
    error("frameDurationが短すぎます。");
end

if L_frm > numAnalysisSamples
    error("フレーム長が解析対象信号の時間長を超えています。");
end

% 端数部分は使用せず、完全に含まれるフレームだけを解析
numFrames = floor( ...
    (numAnalysisSamples - L_frm) / hopLength) + 1;

if numFrames < 1
    error("解析可能なフレームがありません。");
end

fprintf("Sampling frequency      : %.1f Hz\n", fs);
fprintf("Number of channels      : %d\n", m_ch);
fprintf("Analysis duration       : %.3f s\n", round(numAnalysisSamples/fs));
fprintf("Frame length          : %.3f s\n", L_frm);
fprintf("Hop duration            : %.3f s\n", hopLength / fs);
fprintf("Number of frames        : %d\n", numFrames);

%% ---------------------------------------------------------
% 中央時刻に最も近いフレームの決定
% ----------------------------------------------------------

% 各フレームの解析対象信号内での開始サンプル
frameStartSamples = ...
    (0:numFrames-1) * hopLength + 1;

% 各フレームの中央サンプル
frameCenterSamples = ...
    frameStartSamples + (L_frm - 1) / 2;

% 解析対象区間の中央サンプル
analysisCenterSample = ...
    (numAnalysisSamples + 1) / 2;

% 解析対象区間の中央に最も近いフレーム
[~, middleFrameIndex] = min( ...
    abs(frameCenterSamples - analysisCenterSample));

middleFrameStartSample = ...
    frameStartSamples(middleFrameIndex);

middleFrameEndSample = ...
    middleFrameStartSample + L_frm - 1;

middleFrameCenterTime = ...
    round(analysisStartSample/fs) ...
    + (frameCenterSamples(middleFrameIndex) - 1) / fs;

fprintf("Displayed frame index   : %d / %d\n", ...
    middleFrameIndex, numFrames);

fprintf("Displayed frame center  : %.3f s\n", ...
    middleFrameCenterTime);

%% ---------------------------------------------------------
% フレームごとの2D-FFTとパワースペクトル平均
% ----------------------------------------------------------

powerSpectrumSum = [];

middleFrameResult = [];
referenceResult   = [];

for frameIndex = 1:numFrames

    % 現在のフレームのサンプル範囲
    frameStartSample = frameStartSamples(frameIndex);
    frameEndSample   = frameStartSample + L_frm - 1;

    % [時間サンプル × チャンネル]
    frameSignal = x_analy(:, frameStartSample:frameEndSample);

    % 時空間音圧分布と2D-FFTの計算
    currentResult = calculateSpatiotemporalSpectrum_( ...
        frameSignal, fs, micPositions);

    % 複素2Dスペクトルからパワースペクトルを計算
    currentPowerSpectrum = ...
        abs(currentResult.spectrum).^2;

    % パワースペクトルの加算
    if frameIndex == 1

        powerSpectrumSum = ...
            zeros(size(currentPowerSpectrum));

        % 周波数軸などを保存
        referenceResult = currentResult;
    end

    powerSpectrumSum = ...
        powerSpectrumSum + currentPowerSpectrum;

    % 信号中央に最も近いフレームの結果を保存
    if frameIndex == middleFrameIndex
        middleFrameResult = currentResult;
    end
end

%% ---------------------------------------------------------
% パワースペクトルのフレーム平均
% ----------------------------------------------------------

meanPowerSpectrum = ...
    powerSpectrumSum / numFrames;

% 最大値を0 dBとして正規化
maximumPower = max(meanPowerSpectrum, [], "all");

if maximumPower > 0

    meanPowerSpectrumDB = ...
        10 * log10( ...
        max(meanPowerSpectrum / maximumPower, eps));

else

    meanPowerSpectrumDB = ...
        -Inf(size(meanPowerSpectrum));
end

%% ---------------------------------------------------------
% 中央フレームの時間軸
% ----------------------------------------------------------

% calculateSpatiotemporalSpectrumが返す時間軸は
% フレーム先頭を0秒とする相対時間軸であるため、
% 元の録音信号に対する絶対時刻へ変換する。

middleFrameAbsoluteStartTime = ...
    round(analysisStartSample/fs) ...
    + (middleFrameStartSample - 1) / fs;

middleFrameAbsoluteTime = ...
    middleFrameAbsoluteStartTime ...
    + middleFrameResult.time;

%% ---------------------------------------------------------
% 時空間音圧分布画像
% 解析対象信号の中央時刻に最も近い1フレーム
% ----------------------------------------------------------

figure( ...
    "Name", "Spatiotemporal pressure distribution", ...
    "Position", [100, 100, 900, 500]);

imagesc( ...
    middleFrameAbsoluteTime, ...
    middleFrameResult.position, ...
    middleFrameResult.pressureImage);
colormap jet

axis xy;

xlabel("Time [s]");
ylabel("Microphone position [m]");

% title(sprintf([ ...
%     "Spatiotemporal sound-pressure distribution " ...
%     "(frame %d/%d, center = %.3f s)"], ...
%     middleFrameIndex, ...
%     numFrames, ...
%     middleFrameCenterTime));

colorbar;

%% ---------------------------------------------------------
% 時空間2次元スペクトル
% 全フレームの平均パワースペクトル
% ----------------------------------------------------------

figure( ...
    "Name", "Mean spatiotemporal power spectrum", ...
    "Position", [1050, 100, 900, 500]);

imagesc( ...
    referenceResult.frequency / 1000, ...
    referenceResult.spatialFrequency, ...
    meanPowerSpectrumDB);
colormap jet

axis xy;

xlabel("Temporal frequency [kHz]");
ylabel("Spatial frequency [cycles/m]");

% title(sprintf([ ...
%     "Mean spatiotemporal power spectrum " ...
%     "(%d frames, %.1f-s signal)"], ...
%     numFrames, ...
%     analysisDuration));

colorbar;

clim([-displayDynamicRange, 0]);

% %% ---------------------------------------------------------
% % 正の時間周波数だけを表示
% % ----------------------------------------------------------
% 
% positiveFrequencyIndex = ...
%     referenceResult.frequency >= 0;
% 
% figure( ...
%     "Name", ...
%     "Mean spatiotemporal power spectrum: positive frequency", ...
%     "Position", [550, 650, 900, 500]);
% 
% imagesc( ...
%     referenceResult.frequency(positiveFrequencyIndex) / 1000, ...
%     referenceResult.spatialFrequency, ...
%     meanPowerSpectrumDB(:, positiveFrequencyIndex));
% 
% axis xy;

% xlabel("Temporal frequency [kHz]");
% ylabel("Spatial frequency [cycles/m]");
% 
% title(sprintf([ ...
%     "Mean spatiotemporal power spectrum: " ...
%     "positive temporal frequencies " ...
%     "(%d frames)"], ...
%     numFrames));
% 
% colorbar;
% 
% clim([-displayDynamicRange, 0]);