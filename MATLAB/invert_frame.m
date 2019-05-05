function [frame_out] = invert_frame(frame_in)
% INVERT FRAME
% A function that accepts as its input a single frame, inverts pixel 
% intesities and employs a threshold intensity to filter background noise 
% around the worm (epsilon = 0.1), and outputs the resulting frame.
% The function may be adapted to accept intensity threshold as an input.

frame_in = imcomplement(frame_in);
av_bg = mode(frame_in);
range_worm_intensity = max(frame_in(:)) - av_bg;
epsilon = range_worm_intensity * 0.1;

% Two Options:

% 1. MASK - not so good but lets keep the code
% mask = frame_in < av_bg+eps; % ie. not the worm
% frame_in(mask) = 0;
% frame_in = mat2gray(frame_in);

% 2. SUBTRACT
frame_out = frame_in;
frame_out = frame_out - (av_bg+epsilon);
frame_out(frame_out<0) = 0;
frame_out = mat2gray(frame_out);

end %function