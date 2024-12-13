clc
clear 
close all

%% define hyperparameters 
Band = 31;  
upscale_factor = 4;
data_type = 'CAVE';

%% Build directories
currentFolder = pwd;
dataFolder = fullfile(currentFolder, 'Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end

testFolder = fullfile(dataFolder, 'Test', data_type, num2str(upscale_factor));
if ~exist(testFolder, 'dir')
    mkdir(testFolder);
end

validationFolder = fullfile(dataFolder, 'Validation', data_type, num2str(upscale_factor));
if ~exist(validationFolder, 'dir')
    mkdir(validationFolder);
end

%% Source data path
srPath = 'F:/HyperSSR/Test_ablation/train';  % Source data path
srFile = fullfile(srPath);
srdirOutput = dir(fullfile(srFile));
srfileNames = {srdirOutput.name}';
number = length(srfileNames);

%% Limit to folders 21 to 32
startIndex = 21;
endIndex = 31;
srfileNames = srfileNames(startIndex + 2:endIndex + 2); % Adjust for '.' and '..'

%% Loop through selected folders
for index = 1:length(srfileNames)
    name = char(srfileNames(index));
    if (isequal(name, '.') || isequal(name, '..'))
        continue;
    end
    disp(['----:', data_type, '----upscale_factor:', num2str(upscale_factor), '----deal with:', num2str(index), '----name:', name]);

    singlePath = fullfile(srPath, name, name);
    srdirOutput = dir(fullfile(singlePath, '*.png'));
    singlefileNames = {srdirOutput.name}';
    Band = length(singlefileNames);

    % Validate images in folder
    if Band == 0
        warning(['No images found in folder: ', singlePath]);
        continue;
    end

    source = zeros(512 * 512, Band);

    for i = 1:Band
        srName = char(singlefileNames(i));
        srImage = imread(fullfile(singlePath, srName));
        if i == 1
            width = size(srImage, 1);
            height = size(srImage, 2);
        end
        source(:, i) = srImage(:);   
    end

    %% Normalization
    imgz = double(source(:));
    imgz = imgz ./ 65535;
    img = reshape(imgz, width * height, Band);

    %% Obtain HR and LR hyperspectral image
    hrImage = reshape(img, width, height, Band);

    HR = modcrop(hrImage, upscale_factor);
    LR = imresize(HR, 1 / upscale_factor, 'bicubic'); % LR

    %% Save to Test or Validation folders
    if index <= 6
        save(fullfile(testFolder, [name, '.mat']), 'HR', 'LR');
    else
        save(fullfile(validationFolder, [name, '.mat']), 'HR', 'LR');
    end

    clear source HR LR
end

