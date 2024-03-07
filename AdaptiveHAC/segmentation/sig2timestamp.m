function [timestamps] = sig2timestamp(signal,t,method)
%   SIG2TIMESTAMP Extract time stamps from a transition signal
%
%   [tr] = sig2timestamp(S,t) Returns an array of timestamps tr where a
%   transition has happened, based on the change in vector S over a time vector t.
%
%   sig2timestamp(...,method) 'up' (default) gives transition at rising
%   signal, 'nonzero' (used for GT labels) gives transition at nonzero
%   change in signal, 'down' gives transition at falling signal.

%   Author: Nicolas Kruse

arguments
    signal (:,1) {mustBeNumeric}
    t (:,1) {mustBeNumeric}
    method string = 'up'
end

time = linspace(min(t),max(t),length(signal));

if strcmp(method,'up')
    timestamps = time(signal(1:end-1)-signal(2:end)==-1);
elseif strcmp(method,'nonzero')
    timestamps = time(signal(1:end-1)-signal(2:end)~=0);
elseif strcmp(method,'down')
    timestamps = time(signal(1:end-1)-signal(2:end)~=1);
else
    error('%s is not a supported method', method);
end
end