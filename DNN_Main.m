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
[b_N,a_N] = butter(4, [48,52]/fnyq, 'stop');   % it can be tuned
%Bandpass filter
[b_B,a_B] = butter(4, [fcuthigh, fcutlow]/fnyq, 'bandpass');
 
%% Importing raw data from text file and Preprocessing
%NOTE: need to have a folder, in the directory where you run this code,
%called "Delsys", which must have inside it the folders "S1-Delsys-15Class",
%"S2-Delsys-..." etc

Set = dir(".\Delsys")
%this saves, in an array of structures, the names of the folders inside
%this folder


YTemp = categorical(); %array of labels
Ytemp2 = categorical();

k_t = 1;
k_t2 = 1;
L = 4000;
Incr = 2000;
P = 4;

for i = 3:length(Set) %change length(Set) to n+2 to read n subjects' folders
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements) % change length(temp) to n+2 to read n subjects movements .csv files
        name = Movements(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        temp = Rawprocessing(Raw_motion, b_B, a_B, b_N, a_N);
            %buffer2 = Feature_Extr(buffer, N);
        [temp2, N] = Feat_Extr_Overlapp(temp, L, Incr, P);

        YTemp( (k_t-1)*N+1:(k_t*N) , 1) = categorical(sum(double( name)) - offset);
        XTemp( (k_t-1)*N+1:(k_t*N) , :) = temp2;
        k_t = k_t+1;
        
        offset = offset + 1;
        if( offset > 2)
            offset = 0;
        end
        
    end
end
 
     

% Dividing the data set into training and testing/validation and shuffling
% it

[row, col] = size(XTemp);
ind_temp = randperm(row);

XTemp2 =  XTemp(ind_temp, :);
YTemp2 = categorical();
YTemp2(:,1) = YTemp(ind_temp);


divider = ceil(row*0.8)

YTrain = categorical();
YTest = categorical();
XTrain = XTemp2(1:divider, :);
YTrain = YTemp2(1:divider);
XTest = XTemp2(divider+1:row, :);
YTest = YTemp2(divider+1:row);

%% Defining the Network
numFeatures = col;
numClasses = 15;

XValidation = XTest2;
YValidation = YTest2;


layers = [ ...
    featureInputLayer(numFeatures)
    fullyConnectedLayer(numClasses*20)
    sigmoidLayer;
    batchNormalizationLayer;
    dropoutLayer(0.1)
        
    
    fullyConnectedLayer(numClasses*20)
    sigmoidLayer;
    batchNormalizationLayer;
    dropoutLayer(0.1)
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    batchNormalizationLayer;
    dropoutLayer(0.1)
    softmaxLayer
    classificationLayer]

maxEpochs = 120;
miniBatchSize = 60;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold', 1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ... 
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency', 300, ...
    'Verbose',0, ...
    'Plots','training-progress')
    


%% Training

net = trainNetwork(XTrain,YTrain,layers,options);

%% Testing Network

YPred = classify(net, XTest2,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YTest2)./numel(YTest2)

C = confusionmat(YTest2,YPred)
confusionchart(C)
trace(C)

