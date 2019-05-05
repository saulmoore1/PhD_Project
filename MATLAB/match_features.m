function [matched_masked_h5_list, matched_features_list] = match_features(masked_h5_fullfilelist)
% MATCH FEATURES
% A function that takes as its input a list of HDF5/H5 filepaths for masked
% videos (MaskedVideos), loops over the file list to check if corresponding 
% feature file (FeaturesN) exists, and returns a list of filepaths of 
% matched masked videos with their corresponding feature files.

% Initialise empty list for storing names of files for which there exists a
% Teirpsy featuresN results file containing centroid coordinates 
matched_masked_h5_list = {};
if nargout > 1
    matched_features_list = {};
end

for filename = masked_h5_fullfilelist(:)'

    featurefile = replace(char(filename), 'MaskedVideos', 'Results_new');
    featurefile = replace(featurefile, '.hdf5', '_featuresN.hdf5');
    
    if exist(featurefile, 'file') % quicker to explicitly specify that it should look for a file
        matched_masked_h5_list(end+1) = filename; 
        % 'filename' is a 'cell' already, so index with '()' and place at position 'end+1'
        if nargout > 1
            matched_features_list{end+1} = featurefile; 
            % 'featurefile' is a 'char', so '{}' places it inside the cell (position 'end+1')
        end %if
    end %if
end %for
end %function