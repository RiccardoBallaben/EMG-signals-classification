%% Clear all

close all
clear all
clc

%% Initializing the filters

L  = 80000;
fs = 4000;            % sampling frequency [Hz]
f = fs*(0:(L/2))/L;   % frequency resolution from 0 to 1/2 the data length

% Filtering from 20 to 450 Hz
fnyq     = fs/2;        % Nyquist frequency
fcuthigh = 20;          % Highpass cutoff frequency in Hz
fcutlow  = 450;         % Lowpass  cutoff frequency in Hz

%Notch filter
[b_N,a_N] = butter(4, [48,52]/fnyq, 'stop');  
%Bandpass filter
[b_B,a_B] = butter(4, [fcuthigh, fcutlow]/fnyq, 'bandpass');

%% Importing raw data from text file and Preprocessing
%NOTE: need to have a folder, in the directory where you run this code,
%called "Delsys", which must have inside it the folders "S1-Delsys-15Class",
%"S2-Delsys-..." 

Set = dir(".\Delsys")
%this saves, in an array of structures, the names of the folders inside
%this folder.


XTemp = {}; %temporary variable for datas
YTemp = categorical(); %array of labels of datas


pos_index = 1;
L = 4000; %length of window in sample number
M = 100; %number of values taken form each window

Incr = 2000;
for i = 3:length(Set) %change length(Set) to n+3 to read n subjects' folders
    Movements = dir(fullfile(".\Delsys\", Set(i).name) ); 
    offset = 0; %offset used for the conversion of labels into numbers
    
    for j=3:length(Movements) % change length(temp) to n+2 to read n subjects' movements .csv files
        name = Movements(j).name; %name of movement
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        
        %Preprocessing the motion
        temp = Preprocessing(Raw_motion, b_B, a_B, b_N, a_N);
            
        [temp2, S] = Max_Compression(temp, L, Incr); %S = number of segments
            
        k_t = (pos_index-1)*S;
        YTemp(k_t+1 : k_t+S ,1) = categorical(sum(double( name)) - offset);
        XTemp(k_t+1 : k_t+S ,1) = temp2;
        
        pos_index = pos_index+1;
        offset = offset + 1;
        if( offset > 2)
            offset = 0;
        end
        
    end
end

%Shuffling the dataset
ind_temp = randperm(length(XTemp));
XTemp2 = {};
XTemp2(:,1) =  XTemp(ind_temp);
YTemp2 = categorical();
YTemp2(:,1) = YTemp(ind_temp);

%Dividing it into Training and Validation sets
divider = length(XTemp2)*0.8

XTrain = {};
YTrain = categorical();
XTest = {};
YTest = categorical();
XTrain = XTemp2(1:divider);
YTrain = YTemp2(1:divider);
XTest = XTemp2(divider+1:length(XTemp2));
YTest = YTemp2(divider+1:length(YTemp2));

%% Defining the Network

inputSize = 8;
numHiddenUnits = 80;
numClasses = 15; %15 movements

XValidation = XTest;
YValidation = YTest;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    batchNormalizationLayer
    
    fullyConnectedLayer(numClasses*20)
    sigmoidLayer;
    batchNormalizationLayer;
    dropoutLayer(0.2) 
    
    fullyConnectedLayer(numClasses*20)
    sigmoidLayer;
    batchNormalizationLayer;
    dropoutLayer(0.2)
    
    fullyConnectedLayer(numClasses)
    sigmoidLayer
    batchNormalizationLayer
    softmaxLayer
    classificationLayer];  


maxEpochs = 80;
miniBatchSize = 64;

options = trainingOptions('adam', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency', 300, ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...  
    'Verbose',0, ...
    'Plots','training-progress');

%% Training
net = trainNetwork(XTrain, YTrain, layers, options);

%% Testing Network

YPred = classify(net, XTest,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YTest)./numel(YTest)

C = confusionmat(YTest,YPred)

trace(C)

C1 = confusionchart(C);
