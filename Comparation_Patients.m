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

Set = dir(".\Delsys")
%this saves, in an array of structures, the names of the folders inside
%this folder.

k = 1;
range = [7 8] %the patients we want to check
for i = range;
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    for j= 3:4 %the movements we want to check
        Dset(k).folder = Set(i).name;
        Dset(k).name = Movements(j).name;
        temp = table2array(readtable( ...
        fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        Dset(k).sign = Rawprocessing(temp, b_B, a_B, b_N, a_N);
        k = k+1;
    end
end



%% Plotting the motions of the 2 subjects
% Time in seconds
t = (1:1:80000)./4000; % sample/sample frequency

%Patient 5
figure(1)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(1).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 5 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 6
figure(2)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(3).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 6 - Hand closed - Repetition 1 - Sensor 1-8');

%% Plotting the FFT of the 2 motions

L  = 80000;
fs = 4000;            % sampling frequency [Hz]
f = fs*(0:(L/2))/L;   % frequency resolution from 0 to 1/2 the data length

%Patient 5
figure(3)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(1).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 5 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 6
figure(4)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(3).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 6 - Hand closed - Repetition 1 - Sensor 1-8');


%% Plotting the 2 repetitions of subject 5
% Time in seconds
t = (1:1:80000)./4000; % sample/sample frequency

%Patient 5 rep 1
figure(5)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(1).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 5 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 5 rep 2
figure(6)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(2).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 5 - Hand closed - Repetition 2 - Sensor 1-8');

%% Plotting the FFT of the 2 repetitions

L  = 80000;
fs = 4000;            % sampling frequency [Hz]
f = fs*(0:(L/2))/L;   % frequency resolution from 0 to 1/2 the data length

%Patient 5 rep 1
figure(7)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(1).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 5 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 5 rep 2
figure(8)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(2).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 5 - Hand closed - Repetition 2 - Sensor 1-8');



%% Plotting the 2 repetitions of subject 2
% Time in seconds
t = (1:1:80000)./4000; % sample/sample frequency

%Patient 5 rep 1
figure(9)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(3).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 6 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 5 rep 2
figure(10)
for i=1:8
    subplot(3,3,i);
    plot(t,Dset(4).sign(i,:))
    xlabel('Time (s)')
    ylabel('Voltage (V)')
end
sgtitle('Subject 6 - Hand closed - Repetition 2 - Sensor 1-8');


%% Plotting the FFT of the 2 repetitions

L  = 80000;
fs = 4000;            % sampling frequency [Hz]
f = fs*(0:(L/2))/L;   % frequency resolution from 0 to 1/2 the data length

%Patient 5 rep 1
figure(11)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(3).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 6 - Hand closed - Repetition 1 - Sensor 1-8');

%Patient 5 rep 2
figure(12)
for i=1:8
    subplot(3,3,i);
    p2 = fft(Dset(4).sign(i,:));
    p2 = abs(p2/L);
    p2 = p2(1:L/2+1);
    p2(2:end-1)=2*p2(2:end-1);
    
    plot(f, p2)
    xlabel 'Frequency (Hz)'
    ylabel 'Magnitude' 

end
sgtitle('Subject 6 - Hand closed - Repetition 2 - Sensor 1-8');




%% Importing raw data from text file and Preprocessing

Set = dir(".\Delsys")
%this saves, in an array of structures, the names of the folders inside
%this folder.

k = 1;
range = [3] %the patients we want to check
for i = range;
    Movements = dir(fullfile(".\Delsys\", Set(i).name) );
    for j= 3:4 %the movements we want to check
        Dset(k).folder = Set(i).name;
        Dset(k).name = Movements(j).name;
        Dset(k).sign = table2array(readtable( ...
        fullfile(".\Delsys\", Set(i).name, Movements(j).name)));
        %Dset(k).sign = Rawprocessing(temp, b_B, a_B, b_N, a_N);
        k = k+1;
    end
end

%% Plotting the FFT of the HC1 SUBJ 1 REP 1

L  = 80000;
fs = 4000;            % sampling frequency [Hz]
f = fs*(0:(L/2))/L;   % frequency resolution from 0 to 1/2 the data length

p2 = fft(Dset(1).sign(:,3));
p2 = abs(p2/L);
p2 = p2(1:L/2+1);
p2(2:end-1)=2*p2(2:end-1);

figure(1);
plot(f, p2)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'

sig_notch = filter(b_N, a_N, Dset(1).sign(:,3)); %Notch filtering

p3 = fft(sig_notch);
p3 = abs(p3/L);
p3 = p3(1:L/2+1);
p3(2:end-1)=2*p3(2:end-1);

figure(2);
plot(f, p3)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'

sig_band = filter(b_B, a_B, sig_notch); %Bandpass filtering

p4 = fft(sig_band);
p4 = abs(p4/L);
p4 = p4(1:L/2+1);
p4(2:end-1)=2*p4(2:end-1);

figure(3);
plot(f, p4)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'

figure(4)
subplot(3,1,1)
plot(f, p2)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'
title('raw signal')

subplot(3,1,2)
plot(f, p3)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'
title('notch filtered')

subplot(3,1,3)
plot(f, p4)
xlabel 'Frequency (Hz)'
ylabel 'Magnitude'
title('notch and bandpass filtered')

