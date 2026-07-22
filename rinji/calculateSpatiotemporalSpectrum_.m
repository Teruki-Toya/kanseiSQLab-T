%
%   'calculateSpatiotemporalSpectrum_.m'
%       時空間スペクトル解析
%
%	function result = calculateSpatiotemporalSpectrum(
%                     x, fs, micPositions)
%
%	< 入力 >  
%       x:              入力信号 [チャネル数 × 時間サンプル数]
%       fs:             サンプリング周波数 [Hz]
%       micPositions    各マイクロホンの位置 [m]
%
%	< 出力 >
%       result.pressureImage
%           [空間位置 × 時間] の時空間音圧分布
%       result.spectrum
%           複素2次元スペクトル
%       result.spectrumDB
%           最大値を0 dBとして正規化した振幅スペクトル
%       result.time
%           時間軸 [s]
%       result.position
%           空間位置軸 [m]
%       result.frequency
%           時間周波数軸 [Hz]
%       result.spatialFrequency
%           空間周波数軸 [cycles/m]
%	
%	Author:  Teruki Toya
%	Created: Jul. 22, 2026.
%	Copyright: (c) 2026, ASL-UY.
%
function result = calculateSpatiotemporalSpectrum_(x, fs, micPositions)
    arguments
        x double
        fs (1,1) double {mustBePositive}
        micPositions (1,:) double
    end

    %% サイズ確認
    [numChannels, numSamples] = size(x);

    if numChannels ~= numel(micPositions)
        error( ...
            "入力信号のチャンネル数とmicPositionsの要素数が一致しません。");
    end

    if numSamples < 2 || numChannels < 2
        error("時間方向、空間方向ともに2点以上必要です。");
    end

    %% マイクロホン間隔の確認
    positionDiff = diff(micPositions);
    meanSpacing = mean(positionDiff);

    tolerance = 1e-6 * max(1, abs(meanSpacing));

    if any(abs(positionDiff - meanSpacing) > tolerance)
        error([ ...
            "この関数では等間隔アレイを仮定しています。" ...
            "不等間隔アレイの場合は空間方向の補間が必要です。"]);
    end

    %% 時空間音圧分布画像
    % 入力：[チャンネル× 時間]
    % 画像：[空間 × 時間]
    pressureImage = x;

    %% DC成分の除去
    % 各チャンネルの時間平均を除去
    pressureImage = pressureImage ...
        - mean(pressureImage, 2);

    %% 窓関数
    % spatialWindow = win_Hann_(numChannels);
    spatialWindow = ones(1, numChannels);
    temporalWindow = win_Hann_(numSamples).';

    twoDWindow = (temporalWindow * spatialWindow)';

    windowedImage = pressureImage .* twoDWindow;

    %% 2D-FFT
    spectrum = fftshift(fft2(windowedImage));

    magnitudeSpectrum = abs(spectrum);

    %% dB表示
    maximumMagnitude = max(magnitudeSpectrum, [], "all");

    if maximumMagnitude == 0
        spectrumDB = -Inf(size(magnitudeSpectrum));
    else
        normalizedMagnitude = magnitudeSpectrum / maximumMagnitude;

        spectrumDB = 20 * log10( ...
            max(normalizedMagnitude, eps));
    end

    %% 時間軸
    time = (0:numSamples-1) / fs;

    %% 時間周波数軸
    frequency = createCenteredFrequencyAxis(numSamples, fs);

    %% 空間周波数軸
    spatialSamplingFrequency = 1 / meanSpacing;  % [samples/m]

    spatialFrequency = createCenteredFrequencyAxis( ...
        numChannels, spatialSamplingFrequency);

    %% 出力
    result = struct();

    result.pressureImage = pressureImage;
    result.windowedImage = windowedImage;

    result.spectrum = spectrum;
    result.magnitudeSpectrum = magnitudeSpectrum;
    result.spectrumDB = spectrumDB;

    result.time = time;
    result.position = micPositions;
    result.frequency = frequency;
    result.spatialFrequency = spatialFrequency;

    result.samplingFrequency = fs;
    result.microphoneSpacing = meanSpacing;
end


function axisValues = createCenteredFrequencyAxis(numberOfPoints, samplingRate)
% fftshift後に対応する周波数軸を作成する。

    if mod(numberOfPoints, 2) == 0
        index = -numberOfPoints/2 : numberOfPoints/2-1;
    else
        index = -(numberOfPoints-1)/2 : (numberOfPoints-1)/2;
    end

    axisValues = index * samplingRate / numberOfPoints;
end