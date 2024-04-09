function [PBC] = pbc(spectrogram, config_path)
%PBC Compute Power Burst Curve for a spectrogram
%
%   [pbc] = PBC(C) Processes spectrogram C(freq, time, node) and
%   produces Power Burst Curve pbc(time,node).
%   
%   [pbc] = PBC(..., config_path) Loads center frequency band and max
%   frequency bounds from a custom location
%
%   Function can handle both 2D and 3D input (for the 5 monostatic radars).

%   Author: Nicolas Kruse
arguments
    spectrogram
    config_path
end

load(config_path)

for ii = 1:size(spectrogram,3)
    for kk = 1:size(spectrogram,2)
    PBC(kk,ii) = sum(10*log10(abs(spectrogram([1:center_band(1)],kk,ii)).^2)) + ...
        sum(10*log10(abs(spectrogram([center_band(2):end],kk,ii)).^2));
    end
end
end