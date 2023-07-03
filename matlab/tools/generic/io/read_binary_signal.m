function [x, fs, status] = read_binary_signal(ifname)
% read_binary_signal - Reads multichannel signal from a binary file (generated by write_binary_signal)
%
% Syntax: [x, fs, status] = read_binary_signal(ifname)
%
% Inputs:
%   ifname: Input file name (including path and extension)
%
% Outputs:
%   x: Matrix of input data (channels x samples)
%   fs: Sampling frequency
%   status: Status of the file read operation
%           0 - Read successful
%          -2 - File not opened
%          -1 - File size does not match header
%
% Binary file read format:
%   fs (8 bytes), 'double', 'ieee-le'
%   rows (8 bytes), 'uint64', 'ieee-le'
%   cols (8 bytes), 'uint64', 'ieee-le'
%   data (rows x cols) x 8 bytes, 'double', 'ieee-le' (written column-wise)
%
% Revision History:
%   2020: First release
%   2023: Renamed from deprecated version ReadBinaryFile()
%
% References:
%   Reza Sameni, 2020-2023
%   The Open-Source Electrophysiological Toolbox
%   https://github.com/alphanumericslab/OSET

fs = 0; % Initialize the sampling frequency
x = []; % Initialize the data matrix
fileID = fopen(ifname, 'r'); % Open the file for reading
if fileID >= 0
    % Read the header information from the file
    fs = fread(fileID, 1, 'double', 'ieee-le'); % Read sampling frequency (8 bytes)
    rows = fread(fileID, 1, 'uint64', 'ieee-le'); % Read number of rows (8 bytes)
    cols = fread(fileID, 1, 'double', 'ieee-le'); % Read number of columns (8 bytes)
    
    % Read the data from the file
    data = fread(fileID, 'double', 'ieee-le'); % Read data (rows x cols) x 8 bytes, written column-wise
    
    fclose(fileID); % Close the file
    
    if numel(data) == rows * cols
        % Reshape the data to the appropriate dimensions
        x = reshape(data, rows, cols);
        status = 0; % Read successful
    else
        status = -1; % File size does not match header
    end
else
    status = -2; % File not opened
end
