classdef PointCloud
    %POINTCLOUD Class for a Point Cloud Object. Object contains the point
    %data and functions to shuffle, visualise, and normalise the points
    %   Detailed explanation goes here
    
    properties
        data
        PRF = 122
    end
    
    methods
        function obj = PointCloud(data)
            %POINTCLOUD Construct an instance of this class
            %   Detailed explanation goes here
            arguments
                data = NaN
            end
            obj.data = data;
        end
        
        function obj = set.data(obj,data)
            %setData Set the values of the PC
            %   Detailed explanation goes here
            obj.data = data;
        end

        function data = get.data(obj)
            %getData Get the point cloud values
            data = obj.data;
        end

        function obj = normalise(obj,fnc)
            arguments
                obj
                fnc {mustBeA(fnc,'cell')} = {}
            end
            assert(isempty(fnc)||numel(fnc)==size(obj.data,2),"Amount of normalisation function handles (%d) does not correspond to amount of variables (%d) in point cloud",numel(fnc),size(obj.data,2))
            if isempty(fnc)
                obj.data(:,1) = normalize(obj.data(:,1),"center","mean","scale",480);   % range
                obj.data(:,2) = normalize(obj.data(:,2),"scale",obj.PRF);               % doppler
                obj.data(:,3) = normalize(obj.data(:,3),"range");                       % time
                obj.data(:,4) = normalize(obj.data(:,4),"range");                       % power
                obj.data(:,5) = normalize(obj.data(:,5),"scale",5);                     % node
            else
                for n=1:numel(fnc)
                    obj.data(:,n) = fnc{n}(obj.data(:,n));
                end
            end
        end

        function obj = shuffle(obj)
            obj.data = obj.data(randperm(size(obj.data, 1)),:);
        end

        function visualise(obj,options)
        %VISUALISE Visualise Point Cloud
        arguments
            obj
            options.UseCurrentFig = false;
        end
        if ~options.UseCurrentFig; figure; end
        scatter3(obj.data(:,1),obj.data(:,2),obj.data(:,3),[],obj.data(:,4));
        xlabel("Range [#bin]");ylabel("Frequency [#bin]");zlabel("Time [#slice]")
        zticks(1:max(obj.data(:,4)));
        end

    end
end

