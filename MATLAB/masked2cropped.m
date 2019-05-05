function [worm_cropped] = masked2cropped(masked_h5_filepath, stride, SZ, threshold_alive, threshold_time)
% MASKED2CROPPED 
% Accepts a masked video, downsamples uniformly across time by 'stride', 
% and crops to a region centred on the worm (centroid) of size 'SZ'
% REQUIREMENTS: Corresponding featuresN file (to obtain worm coordinates)

%% Input Argument/Error Handling
if nargin < 3 || isempty(SZ)
    SZ = [256, 256]; % Assign default value for crop dimensions
    warning('SZ not specified, using default crop dimensions: 256 x 256')
else %if
    if isscalar(SZ)
        warning('Creating square crop region from scalar input for SZ')
        SZ = [SZ, SZ];
    elseif numel(SZ) > 2
        error('SZ must be a scalar or a vector of max length 2')
    else %if
        if any(SZ) > 2048
            error('Value of SZ too big! Region to crop exceeds original frame size!')
        end %if
    end %if
end %if
        
if nargin < 2 || isempty(stride)
    stride = 5;
    warning('Stride not specified, using default stride: 5')
end %if
    
if nargin < 1 || isempty(masked_h5_filepath)
    warning('No file provided!')
end %if

%% Parameters
% Masked file
dataset = '/mask';
info = h5info(masked_h5_filepath, dataset);
total_frames = info.Dataspace.Size(end);
n_pixels = [info.ChunkSize(1) info.ChunkSize(2)];
n_frames = floor(total_frames./stride);
if mod(total_frames,stride) == 0
    n_frames = n_frames - 1;
end %if

% FeaturesN file (Tierpsy)
features_filepath = replace(masked_h5_filepath, ...
    {'MaskedVideos','.hdf5'}, ...
    {'Results_new','_featuresN.hdf5'});

% Filter featuresN for just the worm trajectory
featuresN = find_worm(features_filepath, threshold_alive, threshold_time);
% find_worm(file, threshold_alive, threshold_time)
if isempty(featuresN)
    worm_cropped = [];
else %if
    % Find worm centroid coordinates
    centroid_coords = round([featuresN.coord_x, featuresN.coord_y]);

    % Preallocate out-array
    worm_cropped = zeros([SZ(1) SZ(2) 1 n_frames]); % already 4D so no need to permute at the end

    % Faster to use for loop than the h5read built-in 'stride' option for downsampling frames
    for fc = 1:n_frames
        % fprintf('%.4d/%.4d', fc, n_frames)
        % Calculate index in original video
        frame_index = (fc-1)*stride+1;

        % Check if featuresN centroid coordinates exist for wanted frame number
        idx_match = (featuresN.frame_number == frame_index);
        if ~any(idx_match)
            % Centroid coordinates do not exist for frame_index in featuresN file for masked_h5_filepath
            % fprintf(repmat('\b',[1,9])) % But why do I need this bit? to keep output tidy
            continue
        end %if
        ind_match = find(idx_match,1); %force it to only take first found frame as match

        % Read indexed frame in masked video 
        % NB: h5read cannot load entire video into 32Gb RAM, as array exceeds max size
        worm_nth_frame = h5read(masked_h5_filepath, dataset, [1 1 frame_index], [n_pixels 1]);

        % Obtain worm centroid coordinates
        worm_centroids = centroid_coords(ind_match,:);

        % Deal with edge issues and out-of-bounds errors
        [x_min, x_max] = deal(worm_centroids(1) - SZ(1)/2, worm_centroids(1) + SZ(1)/2 - 1);
        if x_min <= 0
            [x_min, x_max] = deal(1,SZ(1));
        end %if
        if x_max > n_pixels(2)
            [x_min, x_max] = deal(n_pixels(2)-SZ(1)+1,n_pixels(2));
        end %if
        [y_min, y_max] = deal(worm_centroids(2) - SZ(2)/2, worm_centroids(2) + SZ(2)/2 - 1);
        if y_min <= 0
            [y_min, y_max] = deal(1,SZ(2));
        end %if
        if y_max > n_pixels(1)
            [y_min, y_max] = deal(n_pixels(1)-SZ(2)+1,n_pixels(1));
        end %if

        % Crop to ROI
        worm_nth_frame = worm_nth_frame(x_min:x_max, y_min:y_max);

        % Convert to datatype single and normalise (divide by max uint8)
        worm_cropped(:,:,1,fc) = single(worm_nth_frame)/255;

        % fprintf(repmat('\b',[1,9]))
    end %for
end %if

% Remove empty frames in cropped video (where centroids coords are missing)
idx_frames_to_drop = squeeze(~any(worm_cropped,[1,2]));
worm_cropped(:,:,:,idx_frames_to_drop) = [];

% Clear big matrices from memory
clear featuresN

%% ALTERNATIVELY, manually select ROI to crop
% Find where the worm ever was in the video
% z_projection = max(worm_n_frames,[],3);

% Select manually a region to crop out
% hi = imagesc(z_projection);
% ha = hi.Parent;
% h_roi = drawrectangle(ha);

% Find the extremes
% rect = round(h_roi.Position); %start column (x), start row (y), width, height
% disp(rect)
% rect([3 4]) = max(4*ceil(rect([3 4])./4));
% disp(rect)

% Crop the video (part)
% worm_n_frames = worm_n_frames(rect(2):sum(rect([2,4]))-1,...
%     rect(1):sum(rect([1,3]))-1,:);
% size(worm_n_frames)

% clear z_projection
