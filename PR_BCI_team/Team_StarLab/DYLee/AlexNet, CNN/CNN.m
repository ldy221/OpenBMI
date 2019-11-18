%% Data load
% unzip a image dataset file 

unzip('dataset1.zip');
% loading image data into image data store
imds = imageDatastore('dataset1','IncludeSubfolders',true,'LabelSource','foldernames');

% % dividing the dataset (test data: 70%, validation data: 30%)
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% resize the image to match the input size of the pre-trained network
% input size: 227x227x3
%  3 is the number of colours
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation);
%% define the network architecture
% define the 3 layers convolution neural network architecture
layers1 = [
    imageInputLayer([227 227 3])
   
    % convolution2dLayer(filtersize, numfilter, 'Padding','same')
    convolution2dLayer(3,64,'Padding','same') 
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%% Specify training options
% train the network with an initial learning rate of 0.01 using SGDM(Stochastic Gradient Descent with Momentum)
% set the maximum number of epochs to 4
options1 = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% training the network using the training data 
% Loss is cross entropy losses
% Accuracy is the ratio of images correctly classified by the network
net1 = trainNetwork(augimdsTrain,layers1,options1);

inputSize = net1.Layers(1).InputSize;



%% Displays two sample validation images with predicted labels
YPred = classify(net1,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),2);
 

figure
for i = 1:2
    subplot(1,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
  
%% Calculates the classification accuracy for the validation set
% Accuracy is the percentage of labels that the network correctly predicts
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

