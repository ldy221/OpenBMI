%% using 3 layers CNN to Classify Images 
% read, resize, and classify images using CNN

% use the imread to read the image.
I = imread('face_14.jpg');
figure
imshow(I)

% Returns image B with number of rows and columns specified by vector [numrows numcols] with two elements
 B = imresize(I,[227 227]);
 % use the InputSize property in the first layer of the network to determine the input size of the network
 sz = net1.Layers(1).InputSize;
 
% adjust the image to match the input size of the network
B = B(1:sz(1),1:sz(2),1:sz(3));
figure
imshow(B)

% classify images
label = classify(net1,B)

% Displays the image with the classification result
figure
imshow(B)
title(char(label))
