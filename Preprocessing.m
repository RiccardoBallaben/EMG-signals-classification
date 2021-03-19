function [Motion_out] = Preprocessing(Raw_motion, b_B, a_B, b_N, a_N)
%Takes raw signal and applies notch+bandpass filtering,
%rectification and normalization. Wants coefficients of the Bandpand and
%Notch filter

Motion_out = zeros(8,80000);
for i = 1:8
    temp1 = filter(b_N, a_N, Raw_motion(:,i)); %Notch filtering
    temp2 = filter(b_B, a_B, temp1); %Bandpass filtering
    temp3 = abs(temp2); %Rectification
    Motion_out(i,:) = Sig_norm(temp3)'; %Normalization and transpose
end


end

