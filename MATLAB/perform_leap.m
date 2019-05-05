% MATLAB script to sample frames from cropped/prepared HDF5 quiescence data 
% and facilitate the labelling and training of the LEAP CNN
clear 
close all

%% Parameters
cropped_dir = '/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/';
cropped_list = lookforfile(cropped_dir,'(.*.hdf5)$');
dataset = '/box';
n_videos = 500;
total_frames = 50000;
frames_per_video = total_frames / n_videos;
SZ = [256 256];

fprintf('\nTotal number of files videos to sample for LEAP training data: %d\n', numel(cropped_list));

%% EXTRACT TRAINING FRAMES - uniform sampling (n=5161)
fprintf('\nSelecting %d/%d videos for LEAP training..\n', n_videos, numel(cropped_list));
fprintf('Sampling a total of %d frames..\n\n', total_frames); % to compile a composite video for leap training

sample_list = randsample(cropped_list, n_videos);

% Pre-allocate array for storing extracted frames
training_worms = zeros([SZ(1) SZ(2) 1 total_frames]);
for file_index = 1:numel(sampled_h5_list)
    if mod(file_index,10)==0
        fprintf('Extracted a total of %d/%d frames from %d/%d video files\n', file_index * frames_per_video,...
            total_frames, file_index, numel(sampled_h5_list));
    end %if
    filepath = sampled_h5_list{file_index};
    file = strsplit(filepath,'/');
    file = char(file(end));
    info = h5info(filepath, dataset);
    n_frames = info.Dataspace.Size(end);
    stride = floor(n_frames / frames_per_video);
    % Pre-allocate array each loop iteration to sample n frames from each video
    sampled_frames = zeros([SZ(1) SZ(2) 1 frames_per_video]);
    for fc = 1:frames_per_video
        frame_index = (fc-1)*stride+1;
        frame = h5read(filepath, dataset, [1 1 1 frame_index], [SZ(1) SZ(2) 1 1]);
        sampled_frames(:,:,1,fc) = frame;
    end %for
    % Store sampled frame data in main output array
    training_worms(:,:,1,(file_index-1)*frames_per_video+1:file_index*frames_per_video) = ...
        sampled_frames(:,:,1,:);
end %for

scroll_stack(training_worms)

%% Perform some magic to average the background and invert pixel intensities
% plus thresholding to remove noise in the unmasked region around the worm
size_out = size(training_worms);
n_frames = size_out(4);
for fc = 1:n_frames
    if mod(fc,100) == 0
        fprintf('Performing magic on frame: %d/%d\n',fc,n_frames);
    end %if
    % SET BACKGROUND to average background pixel intensity (lessening masking effect)
    training_worms(:,:,1,fc) = average_background(training_worms(:,:,1,fc));
    % INVERT FRAME to be white worm on black background (for LEAP CNN to work well)
    training_worms(:,:,1,fc) = invert_frame(training_worms(:,:,1,fc));
end %for

scroll_stack(training_worms)

%% Save composite training video for LEAP annotation
out_dir = replace(cropped_dir, 'CroppedVideos', 'TrainingVideos');
if ~exist(out_dir,'dir')
    cmd = ['mkdir -p ', out_dir];
    system(cmd)
end %if
out_filepath = char(join([out_dir,'Training_Frames_n',num2str(total_frames),'.h5'],''));

% Write out in format for LEAP
h5create(out_filepath, dataset, size_out, 'ChunkSize', [size_out(1) size_out(2) 1 1],...
    'Datatype', 'single', 'Deflate', 1)
h5writeatt(out_filepath, dataset, 'dtype', 'single')
h5create(out_filepath, '/ell', [size_out(4) 5], 'Datatype', 'double')
h5writeatt(out_filepath, '/ell', 'dtype', 'double')
h5create(out_filepath, '/framesIdx', [1 size_out(4)], 'Datatype', 'double')
h5writeatt(out_filepath, '/framesIdx', 'dtype', 'double')
h5write(out_filepath, dataset, training_worms, [1 1 1 1], size_out)

%% Clear memory (esp. large matrices) and install LEAP
close all
cd ~/Documents/MATLAB/leap/
install_leap
addpath(genpath('leap'))

% Test Read
% training_worms = h5read("/Volumes/behavgenom$/Saul/singleplatequiescence/TrainingVideos/Training_Frames_n50000_clustersampled.h5","/box");

%% Cluster Sample
% GUI for PCA pre-processing: cluster sampling to select optimal frames for labelling 
cluster_sample 

% Samples_per_video = 10000
% PCs_to_project_data_down_to = 68 (number of principle components that explain 90% variance)
% Use_these_projections_for_clustering = TRUE
% Save as '[x]_clustersampled.h5'

% PCA Warning: Columns of X are linearly dependent to within machine precision.
% Using only the first 15 components to compute TSQUARED.
% First PC appears heavily influenced by the mask shape surrounding the worm, 
% not the worm shape itself, as seen in plotted eigenmodes

%% Create Skeleton
% Import first cluster-sampled frame as first image for labelling
training_worm_path = '/Volumes/behavgenom$/Saul/singleplatequiescence/TrainingVideos/Training_Frames_n50000_clustersampled.h5';
I = h5readframes(training_worm_path, '/box', 1);
create_skeleton % Save as 'worm_skeleton.mat'

%% Label Joints (n=10 body-parts labelled in n=250 PCA pre-processed frames)
% Annotate frames and FAST-TRAIN the network
label_joints

%% Apply trained network to all cropped video files (n=5159)
model_path = '/Volumes/behavgenom$/Saul/singleplatequiescence/TrainingVideos/models/181220_121021-n=250/final_model.h5';
skeleton = load('/Volumes/behavgenom$/Saul/singleplatequiescence/TrainingVideos/worm_skeleton.mat');

% Generate predicted locations for body parts in each video
for file_index = 1%:numel(cropped_h5_fullfilelist)
    %file = '/Volumes/behavgenom$/Saul/singleplatequiescence/TrainingVideos/Training_Frames_n50000_clustersampled.h5';
    file = cropped_list{file_index};
    box = h5read(file, '/box');
    size_out = size(box);
    n_frames = size_out(4);    
    for fc = 1:n_frames
        if mod(fc,500) == 0
            fprintf('Processing frame: %d/%d\n',fc,n_frames);
        end %if
        box(:,:,1,fc) = average_background(box(:,:,1,fc));
        box(:,:,1,fc) = invert_frame(box(:,:,1,fc));
    end %for
    fprintf('Predicting worm body parts in video: %s\n\n',file);
    preds = predict_box(box, model_path); % returns structure containing predicted positions in array of shape:
                                          % (body parts)*(x,y)*(frames), where each row is an image coordinate
end %for

scroll_stack(box)
    
%%
% Visualize Body Part Predictions
% Plot predictions using LEAP helper function and custom video player
imagesc(box(:,:,:,1)), axis image, hold on, plot_joints_single(preds.positions_pred(:,:,1), skeleton)

vplay(box, @(~,idx)plot_joints_single(preds.positions_pred(:,:,idx),skeleton))

% Plot Trajectory of a Single Body Part
figure, plot(squeeze(preds.positions_pred(1,1,:)), squeeze(preds.positions_pred(1,2,:)),'.-')

%%
% A case where Tierpsy failed to track/skeletonize worm (possibly due to low-contrast issues against the background)
% '/Volumes/behavgenom$/Bertie/singleplatequiescence/MaskedVideos/2017_07_01/Bertie_2017_07_01_1/JU258/JU258_Ch2_01072017_094528.hdf5'

% A tricker case same as above
% '/Volumes/behavgenom$/Bertie/singleplatequiescence/MaskedVideos/2017_07_01/Bertie_2017_07_01_2/CX11314/CX11314_Ch2_01072017_100327.hdf5'
