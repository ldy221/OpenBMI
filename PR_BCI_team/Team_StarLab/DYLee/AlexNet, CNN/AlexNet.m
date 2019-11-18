%% Data load
% unzip a image dataset file
unzip('dataset1.zip');
% loading image data into image data store
imds = imageDatastore('dataset1','IncludeSubfolders',true,'LabelSource','foldernames');
% dividing the dataset (test data: 70%, validation data: 30%)
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% resize the image to match the input size of the pre-trained network
% input size: 227x227x3
%  3 is the number of colours
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation);

%% bring the pre-trained Alex Net
net = alexnet;

% check the input size of a first layer
inputSize = net.Layers(1).InputSize;

%% Change the last three layers
% The last three layers of the pre-trained network net are configured for 1000 classes
% So we have to adjust these three layers to fit our new classifier
% Extracts all layers from a pre-trained network, except the last three layers
layersTransfer = net.Layers(1:end-3);

% Converts the last three layers 
% to the fully connected layer, the soft-max layer
% and the categorization output layer to a new classification operation


% 2-class classifier
numClasses = numel(categories(imdsTrain.Labels)) 

layers = [
    layersTransfer
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% training the network
% Specifies the augment operation to be performed further for the training image
% Randomly flip the training image along the longitudinal axis 
% and randomly parallelize up to 30 pixels horizontally and vertically 
pixelRange = [-30 30];
% Automatically resize the training image using the augmented image data store
% augmented data helps to prevent network overfitting and ensure that the exact details of the training images are not remembered.
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% Specify training options
% for transfer learning, maintain the characteristics of the front layer of the pre-trained network 
% (the weight of the transferred layer)
% specifies the mini batch size and verification data
% During training, the network is verified every repeat of ValidationFrequency
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train a network of transfer layers and new layers
netTransfer = trainNetwork(augimdsTrain,layers,options);

%% classifying the validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);

% Displays two sample validation images with predicted labels
idx = randperm(numel(imdsValidation.Files),2);
figure
for i = 1:2
    subplot(1,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% Calculates the classification accuracy for the validation set
% Accuracy is the percentage of labels that the network correctly predicts
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)





