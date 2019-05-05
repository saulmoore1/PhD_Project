function [matches] = lookforfile(target_dir, expr_to_match)
%lookforfile Looks up files in a directory (and subdirectories) that match
%the regular expression expr
%   It will loop on all the subdirectories of target dir. Target dir can be
%   a cell array. matches will be a cell array, N_matches_by_1, with the
%   complete path to the matching files

% We need target_dir to always be a cell array
if ischar(target_dir)
    tmp = target_dir;
    target_dir = cell(1,1);
    target_dir{1} = tmp;
end %if

matches = cell(0,0);

% let's look for matches in all target directories
for dc = 1:numel(target_dir)
    
    %this finds all subdirectories and files in each subdirectory
    [subfold, files] = subdir(target_dir{dc});
    
    % and add the ones in target_dir itself
    subfold{end+1} = target_dir{dc};
    tmp = dir(target_dir{dc});
    files{end+1} = {tmp(~[tmp.isdir]).name};
    
    % for each subdirectory look for match in all possible files
    for sc = 1:numel(subfold)
        
        %local matches in this folder
        loc_match = regexp(files{sc},expr_to_match,'match','once');
        
        idx_full = arrayfun(@(i)~isempty(loc_match{i}),1:numel(loc_match));
        idx_full = find(idx_full);
        
        if ~isempty(idx_full)
            matches(end+1:end+numel(idx_full),1) =...
                arrayfun( @(i)fullfile(subfold{sc},files{sc}{i}),idx_full,'UniformOutput',false);
        end %if
        
    end %for
end %for

end %function



function [subs,fls] = subdir(CurrPath)
%   SUBDIR  lists (recursive) all subfolders and files under given folder
%
%   SUBDIR
%        returns all subfolder under current path.
%
%   P = SUBDIR('directory_name')
%       stores all subfolders under given directory into a variable 'P'
%
%   [P F] = SUBDIR('directory_name')
%       stores all subfolders under given directory into a
%       variable 'P' and all filenames into a variable 'F'.
%       use sort([F{:}]) to get sorted list of all filenames.
%
%   See also DIR, CD

%   author:  Elmar Tarajan [Elmar.Tarajan@Mathworks.de] modified so that it
%   is Unix-friendly by Luigi Feriani [luigi.feriani@gmail.com]
%   version: 2.1
%   date:    16-Feb-2016
%
if nargin == 0
    CurrPath = cd;
end %if
if nargout == 1
    subs = subfolder(CurrPath,'');
else %if
    [subs, fls] = subfolder(CurrPath,'','');
end %if


end %function


function [subs,fls] = subfolder(CurrPath,subs,fls)
%------------------------------------------------
tmp = dir(CurrPath);
tmp = tmp(~ismember({tmp.name},{'.' '..'}));
% tmp = tmp(~ismember({tmp.name},{'..'}));
% write the files in this folder
% fls{end+1} = {tmp(~[tmp.isdir]).name};
% subs{end+1} = fullfile(CurrPath);
for i = {tmp([tmp.isdir]).name}
    subs{end+1} = fullfile(CurrPath,i{:});
    if nargin==2
        subs = subfolder(subs{end},subs);
    else %if
        tmp = dir(subs{end});
        fls{end+1} = {tmp(~[tmp.isdir]).name};
        [subs,fls] = subfolder(subs{end},subs,fls);
    end %if
end %for

end %function