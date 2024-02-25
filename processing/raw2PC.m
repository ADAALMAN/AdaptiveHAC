function [PC] = raw2PC(raw, chunks, npoints, thr, options)
%RAW2PC Summary of this function goes here
%   Detailed explanation goes here
arguments
    raw
    chunks (1,1) = 6
    npoints (1,1) = 1024
    thr (1,1) = 0.8
    options.scaling {mustBeMember(options.scaling, {'equal'})} = 'equal'
    options.nodeProcessing {mustBeNumericOrLogical} = 0
    options.vIselection = 0
end

if ~options.nodeProcessing
    assert(size(raw,3)==1, '3D array entered, but node processing is off.')
end

PC_raw = [];
for iNode = 1:size(raw,3)
    idx = [1:size(raw,2)/chunks:size(raw,2),size(raw,2)];
    for ch = 1:chunks
        dat_split{ch}=raw(:,idx(ch):idx(ch+1),iNode);
        dat_FFT{ch} = db(fftshift(fft(dat_split{ch},[],2),2));
        dat_FFT_bin{ch} = dat_FFT{ch};
        dat_FFT_bin{ch}(dat_FFT{ch}./max(max(dat_FFT{ch}))<thr) = 0;
        dat_FFT_bin{ch}(dat_FFT{ch}./max(max(dat_FFT{ch}))>thr) = 1;
    end
    
    scf = npoints/sum(cellfun(@nnz,dat_FFT_bin));
    for ch = 1:chunks
        dat_FFT_bin_rescaled{ch} = imresize(dat_FFT_bin{ch},sqrt(scf));
        dat_FFT_rescaled{ch} = imresize(dat_FFT{ch},size(dat_FFT_bin_rescaled{ch}));
        [r,v,I] = find(dat_FFT_bin_rescaled{ch});
        t = ones([length(I),1])*ch;
        NodeNo = ones([length(I),1])*iNode;
        P = dat_FFT_rescaled{ch}(sub2ind(size(dat_FFT_rescaled{ch}),r,v));
        PC_raw = [[PC_raw];[r,v,t,P,I,NodeNo]];
end
end
if options.vIselection
    PC_raw(:,7) = options.vIselection*normalize(PC_raw(:,2),"range") + (1-options.vIselection)*normalize(PC_raw(:,5),"range");
    PC_raw = sortrows(PC_raw,7,"descend");
else
    PC_raw = sortrows(PC_raw,5,"descend");
end

if options.nodeProcessing
    PC_raw = PC_raw(1:npoints,[1:4 6]);
else
    PC_raw = PC_raw(1:npoints,1:4);
end
PC = PC_raw;
assert(size(PC,1)==npoints)
end

