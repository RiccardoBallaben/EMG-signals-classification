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

XTemp = {};
YTemp = categorical(); 


k_temp = 1;
L = 4000;
M = 100;
Incr = 2000;


range = [3 4 6 7 8 9 10]
%Reading 6 subjects for the training by changing range variable
for i = range
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements) 
        name = Movements(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        temp = Preprocessing(Raw_motion, b_B, a_B, b_N, a_N);
            
        [temp2, S] = Max_Compression(temp, L, Incr);
            
        k_t = (k_temp-1)*S;
        YTemp(k_t+1 : k_t+S ,1) = categorical(sum(double( name)) - offset);
        XTemp(k_t+1 : k_t+S ,1) = temp2;
        k_temp = k_temp+1;
        
        offset = offset + 1;
        if( offset > 2)
            offset = 0;
        end
        
    end
end

XPatient = {}; %matrix of signals
YPatient = categorical(); %array of labels
k_rob = 1;

for i = 6 %reading movements of remaining subject
    Movements_extra = dir(fullfile(".\Delsys\", Set(i).name) );
    offset = 0;
    for j=3:length(Movements_extra) 
        name = Movements_extra(j).name;
        
        Raw_motion = table2array(readtable( ...
            fullfile(".\Delsys\", Set(i).name, Movements_extra(j).name)));
        temp = Preprocessing(Raw_motion, b_B, a_B, b_N, a_N);
            
        [temp2, S] = MaxUS(temp, L, Incr); 
            
        k_t = (k_rob - 1)*S;
        YPatient(k_t+1 : k_t+S ,1) = categorical(sum(double( name)) - offset);
        XPatient(k_t+1 : k_t+S ,1) = temp2;
        k_rob = k_rob+1;
        
        offset = offset + 1;
        if( offset > 2)
            offset = 0;
        end
        
    end
end


%Shuffling the Training dataset
ind_temp = randperm(length(XTemp));
XTemp2 = {};
XTemp2(:,1) =  XTemp(ind_temp);
YTemp2 = categorical();
YTemp2(:,1) = YTemp(ind_temp);


XTrain = {};
YTrain = categorical();
XTrain = XTemp2;
YTrain = YTemp2;

%% Defining the Network

inputSize = 8;
numHiddenUnits = 80;
numClasses = 15; %15 movements

%Using remaining patient for validation
XValidation = XPatient;
YValidation = YPatient;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    batchNormalizationLayer;
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
net = trainNetwork(XTrain_ran, YTrain_ran, layers, options);

%% Testing Network

YPred = classify(net, XPatient,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YPatient)./numel(YPatient)

C = confusionmat(YPatient,YPred)

C1 = confusionchart(C);

trace(C)


