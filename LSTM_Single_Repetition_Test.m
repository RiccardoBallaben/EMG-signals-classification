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


%% Importing raw data from text file

Set = dir(".\Delsys")

YTemp = categorical(); 
Ytemp_2 = categorical();
XTemp = {}; 
XTemp_2 = {};

k_t = 1;
k_t2= 1;
L = 4000;
Incr = 2000;

for i = 3:length(Set) 
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements) 
        name = Movements(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        temp = Preprocessing(Raw_motion, b_B, a_B, b_N, a_N);
        [temp2, N] = Max_Compression(temp, L, Incr);
            
        %Selecting the replicas to use for validation and training
        if( offset == 0 | offset == 1)
            YTemp( (k_t-1)*N+1:(k_t*N) , 1) = categorical(sum(double( name)) - offset);
            XTemp( (k_t-1)*N+1:(k_t*N) , :) = temp2;
            k_t = k_t+1;
        elseif( offset == 2)
            YTemp_2( (k_t2-1)*N+1:(k_t2*N) , 1) = categorical(sum(double( name)) - offset);
            XTemp_2( (k_t2-1)*N+1:(k_t2*N) , :) = temp2;
            k_t2 = k_t2+1;
        end
        
        offset = offset + 1;
        if( offset > 2)
            offset = 0;
        end
        
    end
end


% Dividing the data set into training and testing/validation and shuffling
% the training set

row = length(XTemp);
ind_temp = randperm(row);
XTemp2 =  XTemp(ind_temp, :);
YTemp2 = categorical();
YTemp2(:,1) = YTemp(ind_temp);
YTrain = categorical();
XTrain = XTemp2;
YTrain = YTemp2;

%% Defining the Network

inputSize = 8;
numHiddenUnits = 80;
numClasses = 15;

XValidation = XTemp_2;
YValidation = YTemp_2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    batchNormalizationLayer;

    fullyConnectedLayer(numClasses*20)
    sigmoidLayer;
    batchNormalizationLayer;
    dropoutLayer(0.2)
    
    fullyConnectedLayer(numClasses)
    sigmoidLayer
    batchNormalizationLayer
    softmaxLayer
    classificationLayer]


maxEpochs = 100;
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

YPred = classify(net, XRobust,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YRobust)./numel(YRobust)

C = confusionmat(YRobust,YPred)

C1 = confusionchart(C);

trace(C)

