clc
clear 
close all
 
%% define hyperparameters 
Band = 31;  
patchSize = 32;
randomNumber = 24;
upscale_factor = 4;
data_type = 'CAVE';
global count
count = 0;
imagePatch = patchSize * upscale_factor;
scales = [1.0, 0.75, 0.5];

%% build upscale folder
currentFolder = pwd;
dataFolder = fullfile(currentFolder, 'Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end
trainFolder = fullfile(dataFolder, 'Train');
if ~exist(trainFolder, 'dir')
    mkdir(trainFolder);
end
caveFolder = fullfile(trainFolder, data_type);
if ~exist(caveFolder, 'dir')
    mkdir(caveFolder);
end
scaleFolder = fullfile(caveFolder, num2str(upscale_factor),'\');
if ~exist(scaleFolder, 'dir')
    mkdir(scaleFolder);
end
savePath = scaleFolder;

%% source data path
srPath = 'F:/HyperSSR/Test_ablation/train';  % Source data downloaded from website 
srFile = fullfile(srPath,'/');
srdirOutput = dir(fullfile(srFile));
srfileNames = {srdirOutput.name}';
number = length(srfileNames) - 2;

%% Limit to first 20 folders
srfileNames = srfileNames(1:22);  % Take only the first 20 folders (+2 for '.' and '..')

%% Loop through folders 1 to 20
for index = 1:20
    name = char(srfileNames(index + 2));  % Adjust index to skip '.' and '..'
    if (isequal(name, '.') || isequal(name, '..'))
        continue;
    end
    disp(['----:', data_type, '----upscale_factor:', num2str(upscale_factor), '----deal with:', num2str(index - 2), '----name:', name]);

    singlePath = fullfile(srPath, name, name);
    srdirOutput = dir(fullfile(singlePath, '*.png'));
    singlefileNames = {srdirOutput.name}';
    Band = length(singlefileNames);
    
    % Validate that there are images to process
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

    %% Ensure width and height are valid
    if width == 0 || height == 0
        error('Width or height of the image is zero. Check the input images.');
    end

    %% normalization
    imgz = double(source(:));
    img = imgz ./ 65535;
    t = reshape(img, width, height, Band);

    %% Process scales
    for sc = 1:length(scales)
        newt = imresize(t, scales(sc));    
        x_random = randperm(size(newt, 1) - imagePatch, randomNumber);
        y_random = randperm(size(newt, 2) - imagePatch, randomNumber);

        for j = 1:randomNumber
            hrImage = newt(x_random(j):x_random(j) + imagePatch - 1, y_random(j):y_random(j) + imagePatch - 1, :);

            label = hrImage;   
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage, 180);  
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage, 90);
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage, 270);
            data_augment(label, upscale_factor, savePath);

            label = flipdim(hrImage, 1);
            data_augment(label, upscale_factor, savePath);
        end
        clear x_random;
        clear y_random;
        clear newt;
    end
    clear t;
end
