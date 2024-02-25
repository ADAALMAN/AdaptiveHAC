function [H] = renyi(spectrogram, alpha)
%   renyi Load spectrogram and compute Rényi entropy
%
%   [H] = renyi(C, alpha) Processes spectrogram C(freq,time,node) and
%   produces renyi entropy H(time,node). Alpha defaults to 0.77
%
%   Function can handle both 2D and 3D input (for the 5 monostatic radars).

%   Author: Nicolas Kruse
arguments
    spectrogram
    alpha = 0.77
end
%% parameters
window_analysis = 1; %amount of time samples in each analysis frame %DEPRECTATED

%% init 
data_FFT_power_norm = zeros(size(spectrogram));
H = zeros([length([window_analysis:size(data_FFT_power_norm,2)]),size(spectrogram,3)]);

%% Renyi computation
%Normalize spectrograms for all 5 nodes, compute entropy
data_FFT_power_norm = spectrogram./sum(spectrogram,1);

for ii = 1:size(spectrogram,3)
    for kk = window_analysis:size(data_FFT_power_norm,2)
        H(kk,ii) = (1/(1-alpha)) * log2(sum(sum(data_FFT_power_norm(:,[kk-(window_analysis-1):kk],ii).^alpha))); %entropy of analysis frame, iterated over spectogram
    end
end
end