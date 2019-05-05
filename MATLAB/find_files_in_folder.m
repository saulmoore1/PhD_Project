function [ filelist ] = find_files_in_folder( regexpstr, folder )
%FIND_FILES_IN_FOLDER finds files whose name contains the string
%regexpstr in the specified folder
% 'folder' can be either a variable or a string. using pwd works as well.
% 'regexpstr' can be a regular expression

% create first list of files
fl = dir(folder);

% find correspondences
idx_match = arrayfun(@(i) ~isempty(regexp(fl(i).name,regexpstr,'once')), (1:numel(fl))');

% find folders
idx_folders = vertcat(fl.isdir);

% create index of found files that aren't folders
idx = idx_match & ~idx_folders;

filelist = fl(idx);


end

