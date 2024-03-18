% Author:       Nicolas Kruse
% E-mail:       n.c.kruse@tudelft.nl
% Affiliation:  TU Delft, Microwave Sensing, Signals and Systems
% 
%
%% 

clear all
clc
load file_names.mat
for i=1:length(namesShort)
    nameShort = namesShort(i); nameLong = namesLong(i);
    folder_path = strcat("M:\ewi\me\MS3\MS3-Shared\Ronny_MonostaticData\",nameLong,"\MAT_data_aligned\");
    for file_path = strcat(file_names,nameShort)
        disp(strcat(folder_path,file_path))
        
        %% Process data set
        [spectrogram,t,f] = process(strcat(folder_path,file_path));
        raw = struct;
        raw.H1 = renyi(spectrogram);
        
        %% init
        H_avg = zeros([size(H,2),size(H,2),size(H,1)]);
        H_score = zeros([size(H,2),size(H,2)]);
        t1 = linspace(min(t),max(t),size(H,1));
        load(strcat(folder_path,file_path,".mat"),'lbl_out');
        
        % GT time stamps
        tr2 = sig2timestamp(lbl_out,t,'nonzero');
        if isempty(tr2)
            continue
        end
        
        %% computing H-score
        for i=1:size(H,2)
            for j=1:size(H,2)
                if i<j
                    H_avg(i,j,:) = mean([H(:,i),H(:,j)],2);
                elseif i == j
                    H_avg(i,j,:) = H(:,i);
                end
                % H-time stamps
                d1 = reshape(H_avg(i,j,:),[],1);
                [~, s1, ~] = lagSearch(d1);
                tr1 = sig2timestamp(s1,t);
                % compute score
                if i <= j; [H_score(i,j), ~] = perfFuncLin(tr1,tr2); end
            end
        end
        
        % 5-node averaged H
        d1 = mean(H,2);
        [~, s1, ~] = lagSearch(d1);
        tr1 = sig2timestamp(s1,t);
        % compute score
        [H_avg_score,~] = perfFuncLin(tr1,tr2);
        
        save(strcat("C:\Users\nckruse\OneDrive - Delft University of Technology\MS3\Segmentation\data\results_TUD\raw\",file_path,".mat"),'H_score','H_avg_score','PBC_score','PBC_avg_score')
    end
end