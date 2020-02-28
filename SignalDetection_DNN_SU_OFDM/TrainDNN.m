%% TrainDNN
%
% This script is to set up parameters for training the deep neural network
% (DNN).

% The DNN is trained for the selected subcarrier based on the training
% data.

%% Clear workspace

clear variables;
close all;

%% Load training and validation data

load('TrainingData.mat');
load('ValidationData.mat');

%% Define training parameters

MiniBatchSize = 1000;
MaxEpochs = 100;
InputSize = 2*NumOFDMsym*NumSC;
NumHiddenUnits = 16;
NumClass = length(Label);

%% Form DNN layers

Layers = [ ...
    sequenceInputLayer(InputSize)
    lstmLayer(NumHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(NumClass)
    softmaxLayer
    classificationLayer];

%% Define trainig options

Options = trainingOptions('adam',...
    'InitialLearnRate',0.01,...
    'ValidationData',{XValid,YValid}, ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'LearnRateDropFactor',0.1,...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',MiniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',0,...
    'Plots','training-progress');

%% Train DNN

Net = trainNetwork(XTrain,YTrain,Layers,Options);

%% Save the DNN

save('TrainedNet','Net','MiniBatchSize');



