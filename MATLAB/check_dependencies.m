function [ dep_list ] = check_dependencies( fun_name )
%check_dependencies Just a wrapper for MATLAB native tool to list the
%dependencies of a script/function

if ~ischar(fun_name)
    error('fun_name has to be an array of characters');
end

dep_list = matlab.codetools.requiredFilesAndProducts(fun_name);
dep_list = dep_list';

end

