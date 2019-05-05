function [featuresN_worm] = find_worm(features_filepath, threshold_alive, threshold_time)
% FIND_WORM
% function to return a truncated featuresN file for 'good worms',
% defined as being a tracked entity (ie. blob), that 
% moves about > threshold no. pixels in the video = classified as a worm
%%
featuresN = h5read(features_filepath, '/trajectories_data');

worm_ids = unique(featuresN.worm_index_joined);
is_a_worm = zeros(1, numel(worm_ids));
for wc = 1:numel(worm_ids)
    id = worm_ids(wc);
    x = featuresN.coord_x(featuresN.worm_index_joined == id);
    y = featuresN.coord_y(featuresN.worm_index_joined == id);
    if numel(x) == numel(y) && ...
            sum(featuresN.worm_index_joined == id) > threshold_time && ...
            ((range(x) > threshold_alive) || (range(y) > threshold_alive))
        is_a_worm(wc) = true;
    end %if
end %for

worm_id = worm_ids(is_a_worm == 1); % return the id of the worm only

%% Filter featuresN and return centroid coords for just the worm
% x_coords_worm = featuresN.coord_x(featuresN.worm_index_joined == worm_id);
% y_coords_worm = featuresN.coord_y(featuresN.worm_index_joined == worm_id);
% numel(x_coords_worm) == numel(y_coords_worm)

%% Filter and return featuresN struct for just the worm
idx = any(featuresN.worm_index_joined == worm_id(:)', 2);
% worm_index_joined is a column, worm_id is forced to be a row == yields a
% many-by-few matrix. By doing any(,2) we take all the instances in which
% worm_index_joined is equal to any of the worm_id

% 1) Use structfun
% tic
featuresN_worm = structfun(@(x) x(idx), featuresN, 'UniformOutput', false);
% toc

% 2) Use FOR-loop + dynamic field names
% tic
% fn = fieldnames(featuresN);
% for i=1:numel(fn)
%     featuresN_worm.(fn{i}) = featuresN.(fn{i})(idx);
% end
% toc

%% Return empty if cropped featuresN struct contains any frame number repeats 
% (ie. fail to find a single set of coordinates for the worm in each frame, so will fail to crop)
if numel(unique(featuresN_worm.frame_number)) ~= numel(featuresN_worm.frame_number)
%     keyboard
    featuresN_worm = [];
end %if
end %function