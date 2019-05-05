function [frame_out] = average_background(frame_in)
% AVERAGE BACKGROUND
% A function that takes as its input a single frame, and returns the frame
% with the mask set to the average background pixel intesity around the worm

mask = frame_in > 0; % logical array of worm region (1) vs masked region (0)
shrunken_mask = imerode(mask, strel('disk',3)); % erode mask region a bit
edge_mask = mask - shrunken_mask; % edge_mask == 1 in area around the worm
av_colour = mean(frame_in(edge_mask == 1)); % calculate mean pixel intensity of this area
frame_in(~mask) = av_colour; % replace mask 0 values (black) with mean value (average colour)
% NB: '~mask' may be faster than 'mask == 0'
frame_out = frame_in;

end % function