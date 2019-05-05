function [hf] = scroll_stack( fs, mask, colormapname, flag_smallfigure )
%scroll_stack Allows to scroll via a slider along the third dimension of a
%3D matrix

if nargin < 4 || isempty(flag_smallfigure)
    flag_smallfigure = false;
end

if nargin < 3 || isempty(colormapname)
    colormapname = 'gray';
end

if nargin < 2 || isempty(mask)
    flag_mask = false;
else
    flag_mask = true;
end

if ndims(fs) > 4 || ndims(fs) < 3
    error('This function only works with 3D or 4D matrices.');
end %if


%% parameters

if ndims(fs) == 4
    flag_4D = true;
else
    flag_4D = false;
end

if flag_4D
    [hh, ww, ~, N_frames] = size(fs);
else
    [hh, ww, N_frames] = size(fs);
end

scrsz = get(0,'ScreenSize');

scrw = scrsz(3); %screen width
scrh = scrsz(4); %screen height


%% GUI

hf = figure;
drawnow;
if ~flag_smallfigure
    maximise_figure(hf);
else
    hf.Position = [hf.Position(1:2)./2, 960, 512];
end

% set scrollwheel callback function

hf.WindowScrollWheelFcn = @scroll_figure;

% position axis
ha = axes('Units','Normalized',...
    'Position',[0 0.05/1.1 1 1.05/1.1]);
ha.Tag = 'ha';

%show first frame
% hi = imshow(fs(:,:,1),[],'Parent',ha);


% gui elements
hs = uicontrol('Style','Slider',...
    'Min',1,'Max',N_frames,...
    'Value',1,...
    'Units','Normalized',...
    'Position',[0 0 1 0.05/1.1],...
    'SliderStep',[1 10]./N_frames);

hl = addlistener(hs,'Value','PostSet',@update_figure);

update_figure;

%% callback functions

    function [] = update_figure(~,~)
        
        
        frame2show = round(hs.Value);
        figure(hf);
        
        if flag_4D
            dummy_frame = fs(:,:,1,frame2show);
        else
            dummy_frame = fs(:,:,frame2show);
        end
        
        if isfloat(fs)
            dummy_frame = mat2gray(dummy_frame);
        end
        
        if ~islogical(fs)
            Low_High = stretchlim(dummy_frame(dummy_frame~=0), [0 1]);
            
            hi = imshow(imadjust(dummy_frame,Low_High),[],'border','tight');
        else
            hi = imshow(dummy_frame,[],'border','tight');
        end %if
        
        % set colormap
        colormap(gca,colormapname);
        
        % add mask
        if flag_mask
            superimpose_a_mask(gca,mask,[1 0 0], 0.3);
        end

        % add frame/slice counter
        set(gca,'units','normalized');
        text(0,0,['frame ',num2str(frame2show),'/',num2str(N_frames)],...
            'Color','g',...
            'FontSize',12,...
            'Units','Normalized',...
            'VerticalAlignment','bottom');


    end



end



function [] = scroll_figure(hObject, callbackdata)


frame2show = round(callbackdata.Source.Children(1).Value + callbackdata.VerticalScrollCount);
frame2show = max(1,frame2show);
frame2show = min(frame2show,callbackdata.Source.Children(1).Max);
callbackdata.Source.Children(1).Value = frame2show;

end



function [  ] = maximise_figure( hf )
%maximise_figure programmatically makes a figure full_screen by mimicking alt+space, x key press
% taken from: http://stackoverflow.com/questions/15286458/automatically-maximize-figure-in-matlab

if nargin < 1 
    hf = gcf;
end %if

if isempty(hf) || ~isvalid(hf)
    warning('hf is empty or not valid, maximising the current figure.');
    hf = gcf;
end %if

figure(hf)                                            %// make hf the current figure
robot = java.awt.Robot; 
robot.keyPress(java.awt.event.KeyEvent.VK_ALT);      %// send ALT
robot.keyPress(java.awt.event.KeyEvent.VK_SPACE);    %// send SPACE
robot.keyRelease(java.awt.event.KeyEvent.VK_SPACE);  %// release SPACE
robot.keyRelease(java.awt.event.KeyEvent.VK_ALT);    %// release ALT
robot.keyPress(java.awt.event.KeyEvent.VK_X);        %// send X
robot.keyRelease(java.awt.event.KeyEvent.VK_X);      %// release X


end