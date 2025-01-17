function [PC] = raw2PC(raw, subsegmentation, param, npoints, thr, time_feature, options,cfg)
%RAW2PC Convert a raw data (range-time) segment to a point cloud
%representation
%   [PC] = raw2PC(RAW, CHUNKS, NPOINTS) Takes input
%   segment RAW with dimensions (range,time,node_number), splits it into
%   CHUNKS subsegments, and converts it into pointcloud PC with NPOINTS
%   total amount of points
arguments
    raw
    subsegmentation = "fixed-amount"
    param (1,1) = 6
    npoints (1,1) = 1024
    thr (1,1) = 0.8
    time_feature = []
    options.normalisation = false
    options.nodeProcessing {mustBeNumericOrLogical} = 0
    options.vIselection = 0
    options.rangeGate {mustBeNumericOrLogical} = 100
    cfg.PRF (1,1) = 122
end

if ~options.nodeProcessing
    assert(size(raw,3)==1, '3D array entered, but node processing is off.')
end 

PC_raw = [];

for iNode = 1:1:size(raw,3)
    %% Make subsegments (indices) based on a fixed amount or length
    if strcmp(subsegmentation, "fixed-amount")
        chunks = param;
        split_idx = int32([1:size(raw,2)/chunks:size(raw,2),size(raw,2)]);      % We split the raw segment into CHUNKS subsegments
    elseif strcmp(subsegmentation, "fixed-length")
        sublength = param;
        for i = 1:int32(size(raw,2)/sublength+1)
            if (i-1)*sublength+1 > size(raw,2)
               split_idx(i-1) = int32(size(raw,2)); % add last index
            else
                split_idx(i) = int32((i-1)*sublength+1); % add index 
            end
        end
    end
    
    for ch = 1:length(split_idx)-1
        %% Split data and generate RD-maps with FFT
        dat_split{ch}=raw(:,split_idx(ch):split_idx(ch+1),iNode);
        if nnz(dat_split{ch})==0; continue; end
        dat_FFT{ch} = db(fftshift(fft(dat_split{ch},[],2),2));              % We take the FFT to create range-Doppler (RD) representations
        f_array = linspace(-cfg.PRF/2,cfg.PRF/2,size(dat_FFT{ch},2))';
        dat_FFT_bin{ch} = dat_FFT{ch};
        dat_FFT_bin{ch}(dat_FFT{ch}./max(max(dat_FFT{ch}))<thr) = 0;        % We create a binary RD map using a fixed threshold
        dat_FFT_bin{ch}(dat_FFT{ch}./max(max(dat_FFT{ch}))>thr) = 1;
        %% center of mass of RD and range Gating
        if options.rangeGate
            cm = CoM(dat_FFT_bin{ch});
            dat_FFT_bin{ch}([1:round(cm(2))-round((480/438)*options.rangeGate) round(cm(2))+round((480/438)*options.rangeGate):end],:)=0;
        end
        %% regionprops
        % the goal is to select the largest continuous region in the RD
        % map, as we assume this originates from our human target
        s = regionprops(logical(dat_FFT_bin{ch}),["Area", "PixelList"]);
        [x,idx]=sort([s.Area],"descend"); s=s(idx);
        if numel(s)>=3; y = vertcat(s(1:3).PixelList); elseif numel(s)==0; continue;  else; y = vertcat(s(1:end).PixelList); end
        v = f_array(y(:,1)); r = y(:,2);
        if strcmp(time_feature, "sequence-based")
            t_size = 1;
            for i = 1:1:ch
                t_size = t_size + (size(dat_split{i}, 2)-1);
            end
            t = ones([size(y,1),1]).*t_size;
        else
            t = ones([size(y,1),1]).*ch;
        end
        NodeNo = ones([size(y,1),1])*iNode;
        P = dat_FFT{ch}(sub2ind(size(dat_FFT{ch}),y(:,2),y(:,1)));
        PC_raw = [[PC_raw];[r,v,t,P,NodeNo]];                               % The point cloud PC has for each point the variables range, doppler, time, intensity, and the node number
    end 
end
if npoints <= size(PC_raw,1); PC_raw = PC_raw(round(linspace(1,size(PC_raw,1),npoints)),:); 
else
    %warning("PC upsampled with %d identical points",npoints-size(PC_raw,1)); 
    PC_raw(end+1:npoints,:) = datasample(PC_raw,(npoints-size(PC_raw,1)),1,"Replace",true);
end
if options.vIselection
    PC_raw(:,6) = options.vIselection*normalize(PC_raw(:,2),"range") + (1-options.vIselection)*normalize(PC_raw(:,5),"range");
    PC_raw = sortrows(PC_raw,6,"descend");
else
    PC_raw = sortrows(PC_raw,4,"descend");
end

%% Normalisation
if options.normalisation
    PC_raw(:,1) = normalize(PC_raw(:,1),'scale',480,'center','mean');           % range
    PC_raw(:,2) = normalize(PC_raw(:,2),'scale',cfg.PRF);    % doppler
    PC_raw(:,3) = normalize(PC_raw(:,3),"range");                               % time
    PC_raw(:,4) = normalize(PC_raw(:,4),"range");                               % power
    PC_raw(:,5) = normalize(PC_raw(:,5),"scale",5);                               % node
end

%% Variable selection

if options.nodeProcessing
    PC_raw = PC_raw(:,1:5);
else
    PC_raw = PC_raw(:,1:5);
end
PC = PC_raw(:,:);

%% PC checks
assert(size(PC,1)==npoints)

function ct = CoM(A)
    x = 1 : size(A, 2);
    y = 1 : size(A, 1); 
    [X, Y] = meshgrid(x, y);
    meanA = mean(A(:));
    ct(1) = mean(A(:) .* X(:)) / meanA;
    ct(2) = mean(A(:) .* Y(:)) / meanA;
end

end

