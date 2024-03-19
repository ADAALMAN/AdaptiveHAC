function [score, used_trans] = perfFuncLin(trans_test, trans_GT)
%perfFuncLin Use linear scoring to evaluate the quality of found transitions
%   [score, used_trans] = perfFuncLin(x,gt) returns a scoring for the similarity between a
%   sequence x and a ground truth gt. Both x and gt should be sorted 1D
%   vectors containing time stamps.

%   Author: Nicolas Kruse
arguments
    trans_test (:,1)
    trans_GT (:,1)
end

%% init
assert(~isempty(trans_GT),'No ground truth transitions given')
window_size = 2;
dist_matrix = zeros ( [length(trans_test), length(trans_GT)] );
used_trans = []; unused_trans = trans_test;
score = 0;
if isempty(trans_test)
    return
end

%% computation
for i = 1:length(trans_GT)
    dist_matrix(:,i) = abs( trans_GT(i)-trans_test );
end

while ~isempty(dist_matrix)
    delay = min(min(dist_matrix));
    [row, col] = find( dist_matrix==delay , 1 );
    used_trans = [used_trans, unused_trans(row)];
    unused_trans(row) = [];
    dist_matrix(row,:) = []; dist_matrix(:,col) = [];
    if delay < window_size
        score = score + 1; %-delay./window_size; %remove comment to go to non-trapezoid
    elseif delay >= window_size && delay < 1.5*window_size %trapezoid with width (window_size) and falling slope of length (window_size/2)
        score = score + 1-(delay-window_size)./(window_size/2);
    end
end
score = score./max([length(trans_test),length(trans_GT)]);
end