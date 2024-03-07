function [y,signal,threshold] = lagSearch(x,lag,threshold)
%lagSearch Finite difference for a given lag
%   lagSearch(x) finds the finite difference for time series x with a
%   default lag of 47 and outputs a signal when the difference exceeds
%   std(x)
%
%   lagSearch(...,lag) finds finite difference for given lag
%
%   lagSearch(...,threshold) sets a custom threshold
%
%   [y,signal,threshold] = lagSearch(...) returns finite difference,
%   thresholded signal, and threshold
arguments
    x (:,1)
    lag (1,1) {mustBeInteger} = 47
    threshold (1,1) = std(x,'omitnan')
end
y = padarray(x(1:end-lag)-x(lag+1:end),[lag,0],0,'pre');
signal = zeros(length(x),1); signal(abs(y)>threshold) = 1;
end

