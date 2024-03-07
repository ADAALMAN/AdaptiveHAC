function visPC(PC,options)
%VISPC Visualise Point Cloud
%   Detailed explanation goes here
arguments
    PC
    options.UseCurrentFig = false;
end
if ~options.UseCurrentFig; figure; end
scatter3(PC(:,1),PC(:,2),PC(:,3),[],PC(:,4));
xlabel("Range [#bin]");ylabel("Frequency [#bin]");zlabel("Time [#slice]")
zticks(1:max(PC(:,4)));
end

