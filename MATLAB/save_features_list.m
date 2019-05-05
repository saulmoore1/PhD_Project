% MALTAB script to save featuresN file path list of Priota's Food Choice 
% Assay data to a text file, to be read in by 'trajectory_plots.py'

features_dir = '/Volumes/behavgenom$/Priota/Data/FoodChoiceAssay/Results';
features_list = lookforfile(features_dir,'(.*_featuresN.hdf5)$');

out_dir = '/Volumes/behavgenom$/Saul/FoodChoiceAssay/';

% Save featuresN filepath list to text file
if ~exist(out_dir,'dir')
    cmd = ['mkdir -p ', out_dir];
    system(cmd)
end %if

out_filepath = [out_dir, 'FeaturesFilePathList.txt'];
if exist(out_filepath, 'file')
    delete(out_filepath)
end %if

fileID = fopen(out_filepath,'w');
fprintf(fileID, '%s\n', string(features_list));
fclose(fileID);