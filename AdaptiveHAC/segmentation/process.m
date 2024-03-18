function [data_FFT_power, t, f] = process(data, config_path)
%   PROCESS Load raw data and produce spectrogram
%
%   [data_FFT_power, t, f] = PROCESS(data_path) Loads data with name
%   '[data_path].mat' and outputs spectrogram(s) with associated time and
%   frequency values
%
%   PROCESS(..., config_path) uses non-default path for config folder
%
%   Function can handle both 2D and 3D input (for the 5 monostatic radars).

%   Author: Adapted from R. Guendel by Nicolas Kruse
arguments
    data = [];
    config_path = [pwd '\Segmentation\data\config_monostatic_TUD.mat'];
end

data_raw = data;

% Spectrogram
% Parameters
load(config_path)
sample_rate = 1/sample_time;
for kk = 1:size(data_raw,3)
    data_FFT(:,:,kk) = fft(data_raw(:,:,kk));
    [S_M2(:,:,kk), f, t, data_FFT_power(:,:,kk)] = spectrogram(detrend(data_FFT(centerbin,:,kk),2),hann(window_size),window_overlap,nfft,sample_rate,'yaxis','centered','MinThreshold',min_thres);
end
toc
end