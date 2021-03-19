function [Motion_out_raw] = Rawprocessing(Raw_motion, b_B, a_B, b_N, a_N)
%Takes raw signal and applies notch+bandpass filtering.
%Wants coefficients of the Bandpass and Notch filter

Motion_out_raw = zeros(8,80000);
for i = 1:8 % for each sensor
    temp1 = filter(b_N, a_N, Raw_motion(:,i)); %Notch filtering
    Motion_out_raw(i,:) = filter(b_B, a_B, temp1); %Bandpass filtering
end