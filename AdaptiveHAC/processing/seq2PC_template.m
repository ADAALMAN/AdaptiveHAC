clear
%% Parameters
window = 256;
npoints = 1024;
chunks = 6;
vars = [1:4 6]; %variables to save to txt file

data_path_base = ['M:\ewi\me\MS3\MS3-Shared\Ronny_MonostaticData\*\*\*mon*'];
save_path = ['C:\Users\nckruse\OneDrive - Delft University of Technology\MS3\PC\scripts\new_pipeline\temporal\test\'];

%% Init
activities = {'na','wlk','stat','sitdn','stupsit','bfrsit','bfrstand','ffw','stup','ffs'};
counters = ones([size(activities)]);
fail_counter = 0;

file_struct = dir(data_path_base);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% To exclude: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_struct = file_struct(~contains({file_struct.folder},'Nicolas'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_list = {};

wb = waitbar((1/(numel(file_struct))),"Calculating remaining time...");

for act=1:numel(activities); mkdir(fullfile(save_path, 'data', activities{act})); end
seq(numel(file_struct)) = struct();
%% Main Loop
for i = 1:numel(file_struct)
    tic
    file_path = fullfile(file_struct(i).folder,file_struct(i).name);
    L = load(file_path); data = L.hil_resha_aligned; lbl_out = L.lbl_out;
    if length(lbl_out) > size(data,2); lbl_out = lbl_out(1:size(data,2)); end
    
    lbl = 0;

    tmp = split(file_path,'\'); seq(i).filename = tmp{end};

    c = mat2cell(data,480,diff([0:window:length(lbl_out)-1,length(lbl_out)]),5);
    cl = mat2cell(lbl_out,1,diff([0:window:length(lbl_out)-1,length(lbl_out)]));

    t_start=[0:window:length(lbl_out)-1,length(lbl_out)]+1;t_start=t_start(1:end-1);
    t_end = [0:window:length(lbl_out)-1,length(lbl_out)];t_end=t_end(2:end);

    seq(i).sample(numel(c)-1) = struct();
    for k = 1:numel(c)-1
        %% Compute PCs
        PC = PointCloud;
        seq(i).sample(k).timespan = [t_start(k) t_end(k)];
        for n=1:5
            PC(n) = PointCloud(raw2PC(c{k}(:,:,n),chunks,npoints,"normalisation",false));
        end
        if any(cell2mat(cellfun(@(x)any(isnan(x)),{PC.data},'UniformOutput',false)));
            fail_counter=fail_counter+1; continue; end
        %% determine label
        lbl = mode(cl{k})+1;
        if lbl==1; continue; end % don't save samples with label 'na' to save time
        %% save matrices
        for n=1:5
            writematrix(PC(n).normalise.data(:,vars),[save_path 'data\' lower(activities{lbl}) '\' lower(activities{lbl}) '_' num2str(counters(lbl)) '.txt']);
            file_list{end+1,1} = [lower(activities{lbl}) '/' lower(activities{lbl}) '_' num2str(counters(lbl))];
            seq(i).sample(k).final_name(n) = string([lower(activities{lbl}) '/' lower(activities{lbl}) '_' num2str(counters(lbl))]);
            counters(lbl) = counters(lbl)+1; 
        end

    end
    elapsedTime(i) = toc;
    waitbar((i/(numel(file_struct))),wb,[num2str(i) '/' num2str(numel(file_struct)) '. Remaining time: ~' num2str(round(((numel(file_struct)-i)*mean(elapsedTime))/60)) 'min'],"Name",'Progress')
end
writecell(file_list,fullfile(save_path,'data','train_set.txt'));
save(fullfile(save_path,'data','seq_info_train.mat'),'seq')
close(wb)
beep