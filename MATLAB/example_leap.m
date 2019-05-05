% Example LEAP Tutorial

% Clone repository & install LEAP
!git clone https://github.com/talmo/leap.git
cd leap/
install_leap
addpath(genpath('leap'))
%%
% Inspect examplar HDF5 video (cat)
h5disp('box.h5') % read '.h5' file
box = h5read('box.h5','/box');
size(box) % 192x192x1 (2085 frames)
imagesc(box(:,:,:,1)),axis image,colorbar % first frame of HDF5
%%
% GUI PCA cluster sampling to select 'representative' frames for labelling 
cluster_sample
%%
% Once selected frames saved as '.h5', annotate skeleton in ~10 frames
create_skeleton
%%
% Import first cluster-sampled frame as first image for labelling
I = h5readframes('cat_for_labelling.h5','/box',1);
%%
% Labelling and Training Step (label as few as ~10 frames)
% Prompted to choose two files
% 1. Select '.h5' file -- output from cluster_sample
% 2. Select initial annotated frame '.mat' (skeleton) file -- output from 
% create_skeleton
label_joints
% If above gui fails, do:
% pyversion /Users/sm5911/anaconda3/bin/python

%%
% After training epochs, load original movie and generate predicted
% locations for each frame
modelPath = '/Users/sm5911/Documents/MATLAB/models/181016_151248-n=21/final_model.h5';
preds = predict_box(box, modelPath);
% outputs array of shape: (body parts) x (x,y) x (frames)
% where each row is an image coordinate

% Quick visualization of predictions -- plotting with our helper function
skeleton = load('cat_skeleton_for_labelling.mat');
imagesc(box(:,:,:,1)),axis image,hold on,plot_joints_single(preds.positions_pred(:,:,1),skeleton)

% Visualisation using built-in custom video player
vplay(box, @(~,idx)plot_joints_single(preds.positions_pred(:,:,idx),skeleton))

% Plotting trajectory of a single body part (pulling out a single row)
figure,plot(squeeze(preds.positions_pred(3,1,:)),squeeze(preds.positions_pred(3,2,:)),'.-')

% Happy LEAPing!