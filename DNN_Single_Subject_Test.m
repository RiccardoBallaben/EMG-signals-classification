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

Set = dir(".\Delsys")


YTemp = categorical(); 
Ytemp2 = categorical();

k_t = 1;
k_t2 = 1;
L = 4000;
Incr = 2000;
P = 4;
range = [4 5 6 7 8 9 10]; %reading 6 subjects for the training
for i = range 
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements) 
        name = Movements(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        temp = Rawprocessing(Raw_motion, b_B, a_B, b_N, a_N);
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

for i = 3
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements) 
        name = Movements(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        temp = Rawprocessing(Raw_motion, b_B, a_B, b_N, a_N);
        [temp2, N] = Feat_Extr_Overlapp(temp, L, Incr, P);

        YTemp_2( (k_t2-1)*N+1:(k_t2*N) , 1) = categorical(sum(double( name)) - offset);
        XTemp_2( (k_t2-1)*N+1:(k_t2*N) , :) = temp2;
        k_t2 = k_t2+1;
        
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

YTrain = categorical();
YTest = categorical();
XTrain = XTemp2;
YTrain = YTemp2;



%% Defining the Network
numFeatures = col;
numClasses = 15;

XValidation = XTemp_2;
YValidation = YTemp_2;

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
    classificationLayer]

maxEpochs = 150;
miniBatchSize = 63;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ... 
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency', 300, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Training

net = trainNetwork(XTrain,YTrain,layers,options);

%% Testing Network

YPred = classify(net, XTest2,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YTest2)./numel(YTest2)

C = confusionmat(YTest2,YPred)
confusionchart(C)
trace(C)
