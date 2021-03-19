function [Max_motion, N] = Max_Compression(Motion, L, Incr)
%INPUTS:
%Motion = the 8 sensors of a motion after the Preprocessing()
%L = length of window of time given as number of samples, 1 sec -> L = 4000
%Incr = how many samples the window slides each time (Overlapping windows)

N = (80000 - (L-Incr))/Incr; %number of segments, which overalpp of Incr samples

L_s = L/100; %number of samples of each sub-interval in each segment,
% assuming we extract 100 values form each window


Max_motion = cell(N,1);
temp_ = zeros(8,100);
for j=1:N %selects the segment of 1 second
    for i=1:8 %selects the sensor
        for k=1:100 %selects the sub-intervals
            m = max(Motion(i, (j-1)*Incr+1 + (k-1)*L_s : (j-1)*Incr + k*L_s) );
            temp_(i, k) = m;
        end
    end
    Max_motion(j) = {temp_};
end

end

