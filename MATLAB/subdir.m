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
end% if
if nargout == 1
   subs = subfolder(CurrPath,'');
else
   [subs fls] = subfolder(CurrPath,'','');
end% if
  %
  %
function [subs,fls] = subfolder(CurrPath,subs,fls)
%------------------------------------------------
tmp = dir(CurrPath);
tmp = tmp(~ismember({tmp.name},{'.' '..'}));
for i = {tmp([tmp.isdir]).name}
   subs{end+1} = fullfile(CurrPath,i{:});
   if nargin==2
      subs = subfolder(subs{end},subs);
   else
      tmp = dir(subs{end});
      fls{end+1} = {tmp(~[tmp.isdir]).name};
      [subs fls] = subfolder(subs{end},subs,fls);
   end% if
end% if