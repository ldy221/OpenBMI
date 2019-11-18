%% using AlexNet to Classify Images
% read, resize, and classify images using AlexNet

% use the imread to read the image
I = imread('face_0.jpg');
figure
imshow(I)

% Returns image B with number of rows and columns specified by vector [numrows numcols] with two elements
 B = imresize(I,[227 227]); 
 % use the InputSize property in the first layer of the network to determine the input size of the network
 sz = netTransfer.Layers(1).InputSize;
 
% adjust the image to match the input size of the network
B = B(1:sz(1),1:sz(2),1:sz(3));
figure
imshow(B)

% classify images
label = classify(netTransfer,B)

% Displays the image with the classification result
figure
imshow(B)
title(char(label))
