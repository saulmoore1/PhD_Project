% MATLAB script to prepare Bertie's quiescence masked video data for LEAP
% Crops, prepares and checks HDF5 videos for compatibility with LEAP labelling and training
clear
close all

%% Parameters
masked_dir = '/Volumes/behavgenom$/Bertie/singleplatequiescence/MaskedVideos/';
out_dataset = '/box'; % Necessary for LEAP GUI (MATLAB add-on app)
stride = 5;
SZ = [256 256];
threshold_alive = 10; % NB: 1 pixel = 10 microns
threshold_time = 100; % NB: 25 frames = 1 second

masked_list = lookforfile(masked_dir,'(.*.hdf5)$'); % returns cell array (n=5793)
disp(['Number of masked video files found: ', num2str(numel(masked_list))]);

matched_masked_list = match_features(masked_list);

NoFeatureFiles = setdiff(masked_list,matched_masked_list); % videos without featuresN (n=5)
disp([num2str(numel(NoFeatureFiles)), ' files were removed because no matching featuresN file was found']);
disp(['Number of files to crop: ', num2str(numel(matched_masked_list'))])

%% Crop Videos (n=5788)
error_log_path = '/Volumes/behavgenom$/Saul/singleplatequiescence/ErrorLogs/crop_error_log.txt';
fid = fopen(error_log_path, 'w');

% This for-loop may be run in parallel instances of MATLAB
excluded_files = {};
for file_index = 1:numel(matched_masked_list)
    try
        % fprintf('%.4d/%.4d', file_index, numel(matched_masked_h5_list'))
        disp(['Cropping file: ',num2str(file_index),'/',num2str(numel(matched_masked_list')),' (',num2str(file_index/numel(matched_masked_list')*100),'%)'])
        % Select masked HDF5 video file from the archive shared folder
        masked_filepath = matched_masked_list{file_index}; 
        
        % Downsample, crop, permute and normalize the HDF5 video
        worm_cropped = masked2cropped(masked_filepath, stride, SZ, threshold_alive, threshold_time);

        if isempty(worm_cropped)
            excluded_files{end+1} = masked_filepath;
            fprintf(fid, 'Failed to crop file: %s \nFrame number repeats in featuresN file. \nCannot extract centroid coordinates for target worm.\n\n', masked_filepath);
        else %if
            % Output cropped video to my directory in the new shared folder
            out_filepath = replace(masked_filepath, ...
                                        {'Bertie','MaskedVideos','.hdf5'}, ...
                                        {'Saul','CroppedVideos','_cropped.hdf5'});
            out_dir = strsplit(out_filepath,'/');
            out_dir = char(join(out_dir(1:end-1), '/'));
            % Re-create the same directory structure as in Bertie's behavgenom_archive$ folder
            if ~exist(out_dir,'dir')
                cmd = ['mkdir -p ', out_dir];
                system(cmd)
            end %if
            if exist(out_filepath, 'file')
                delete(out_filepath)
            end %if
            Size_ROI = size(worm_cropped);
            h5create(out_filepath, out_dataset, Size_ROI, 'ChunkSize', [Size_ROI(1) Size_ROI(2) 1 1],...
                'Datatype', 'single', 'Deflate', 1)
            h5writeatt(out_filepath, out_dataset, 'dtype', 'single')
            h5create(out_filepath, '/ell', [Size_ROI(4) 5], 'Datatype', 'double')
            h5writeatt(out_filepath, '/ell', 'dtype', 'double')
            h5create(out_filepath, '/framesIdx', [1 Size_ROI(4)], 'Datatype', 'double')
            h5writeatt(out_filepath, '/framesIdx', 'dtype', 'double')
            % Write out to new HDF5 array, reshaped to include dimension for RGB channel
            h5write(out_filepath, out_dataset, worm_cropped, [1 1 1 1], Size_ROI)
        end %if
    catch EE
        fprintf(fid, 'Failed to crop video: %s\n', masked_filepath);
        fprintf(fid, '%s\n\n', EE.message);
    end %try
    % fprintf(repmat('\b',[1,9]))
end %for

fclose(fid);
fprintf('\n\nCOMPLETE!\n Number of files cropped: %d\n Number of files excluded: %d/%d\n\n', numel(matched_masked_list) - numel(excluded_files), numel(excluded_files), numel(matched_masked_list'));
% Clear big matrices from memory
clear worm_cropped

% 306+168+140 = 614
% excluded_files = 614 (due to multiple frame instances, ie. multiple coords for the worm)

%% Check Videos (n=5174)
cropped_dir = '/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/';
cropped_list = lookforfile(cropped_dir,'(.*.hdf5)$');
fprintf('Number of cropped videos found: %d\n\n', numel(cropped_list));
dataset = '/box';

masked_files = {};
for fc = 1:numel(matched_masked_list)
    file = strsplit(matched_masked_list{fc}, '/');
    file = file{end};
    masked_files{end+1} = file;
end %for

cropped_files = {};
for fc = 1:numel(cropped_list)
    file = strsplit(cropped_list{fc}, '/');
    file = file{end};
    file = join(strsplit(file, '_cropped'),'');
    cropped_files(end+1) = file;
end %for

failed2crop = setdiff(masked_files, cropped_files);
fprintf('%d masked videos failed to crop.\n\n', numel(failed2crop));

%% FLAG EMPTY VIDEOS (any videos containing empty frames) (n=0)
empty_video_log = '/Volumes/behavgenom$/Saul/singleplatequiescence/ErrorLogs/empty_video_log.txt';
fid = fopen(empty_video_log,'w');

bvc = 0;
disp('Looking for EMPTY VIDEOS...');
for fc = 1:numel(cropped_list)
    if mod(fc,10)==0
        fprintf('%d/%d\n', fc, numel(cropped_list));
    end %if
    try
        worm_cropped = h5read(cropped_list{fc}, '/box');
    catch EE
        fprintf(fid, 'CANNOT READ video file %d: %s \n', fc, cropped_list{fc});
        fprintf(fid, '%s\n\n', EE.message);
        disp(['CANNOT READ: ' cropped_list{fc}]);
        cropped_list(fc,:) = [];
        continue
    end %try
    bad_video = ~all(squeeze(any(any(worm_cropped,1),2))); % check if any frames are empty in the video
    if bad_video
        fprintf(fid, 'EMPTY VIDEO. Empty frames in file %d: %s \n', fc, cropped_list{fc});
        disp(['EMPTY VIDEO: ' cropped_list{fc}]);
        cropped_list(fc,:) = [];
        bvc = bvc + 1;
    end %if
end %for

fclose(fid);
disp(['Number of EMPTY VIDEOS found: ', bvc]);

%% FILTER BAD VIDEOS via MANUAL INSPECTION (n=5158, bad=15, missing=1)
% On inspection, some videos that were successfully cropped contained significant
% moving and persisting 'blobs' (not worms) that filtered through my
% thresholding and were mis-identified as the worm by my function find_worm()
% These cases were manually detected, and confirmed by inspecting the original masked videos (in Tierpsy)

% Videos were omitted if it was confirmed that:
% 1. The real worm eluded the Tierpsy tracker and was not skeletonised
% 2. The real worm does not move much in the video, thereby not providing all that rich / phenotypically varied data

videos2delete = {
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_08_22/Bertie_2017_08_22_3/NIC265/NIC265_Ch2_22082017_105251_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_09_19/Bertie_2017_09_19_3/ECA372/ECA372_Ch1_19092017_111129_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_04/Bertie_2017_07_04_2/JU847/JU847_Ch1_04072017_131624_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_16/Bertie_2017_07_16_3/JU2001/JU2001_Ch1_16072017_101152_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_08_16/Bertie_2017_08_16_3/NIC207/NIC207_Ch2_16082017_102753_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_06_21/Bertie_2017_06_21_3/DL238/DL238_Ch2_21062017_114003_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_16/Bertie_2017_07_16_2/JU1586/JU1586_Ch2_16072017_090850_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_28/Bertie_2017_07_28_3/NIC242/NIC242_Ch1_28072017_111315_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_08_16/Bertie_2017_08_16_3/QG557/QG557_Ch2_16082017_101145_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_01/Bertie_2017_07_01_2/JU258/JU258_Ch1_01072017_102017_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_09_20/Bertie_2017_09_20_3/XZ1516/XZ1516_Ch2_20092017_095524_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_09_23/Bertie_2017_09_23_3/MY2212/MY2212_Ch1_23092017_100926_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_06_21/Bertie_2017_06_21_3/JU775/JU775_Ch2_21062017_131853_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_07_14/Bertie_2017_07_14_1/JT11398/JT11398_Ch1_14072017_121949_cropped.hdf5'},...
{'/Volumes/behavgenom$/Saul/singleplatequiescence/CroppedVideos/2017_06_10/Bertie_2017_06_10_1/Haw/Haw_Ch1_10062017_110417_cropped.hdf5'}};

bad_video_log = '/Volumes/behavgenom$/Saul/singleplatequiescence/ErrorLogs/bad_video_log.txt';
fid = fopen(bad_video_log,'w');
for i = 1:numel(videos2delete)
    if exist(videos2delete{i}{1}, 'file')
        fprintf('Deleting bad video: %s\n', videos2delete{i}{1});
        cmd = ['rm ', videos2delete{i}{1}];
        system(cmd)
    end %if
    fprintf(fid, 'DELETED BAD VIDEO: %s\n\n', videos2delete{i}{1});
end %for

fclose(fid);
fprintf('\n%d videos were manually deleted.\n', numel(videos2delete));
