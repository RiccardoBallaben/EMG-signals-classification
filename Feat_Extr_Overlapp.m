function [Feats, N_seg] = Feat_Extr_Overlapp(Motion, L, Incr, P)
%FEAT_EXTR_OVERLAPP extracts features using a sliding window of time
% P = order of AR model, L = number of samples of window, Incr = increment

% Testing

% for m = 1:8
%     Motion(m,:) = 1:80000;
% end
% L = 2000;
% Incr = 1000;
% P = 4;



N = (80000 - (L-Incr))/Incr; %number of segments, which overalp of Incr samples
N_seg = N;

Rms = zeros(8, N);
Mav = zeros(8, N);
Wamp = zeros(8, N);
C = P+1; %the number of coefficients from a P order model is P+1
Ar = zeros(8,C*N);
Wl = zeros(8, N);
T = 5e-4; %Threshold value for the Willison Amplitude features

for i=1:8
   for j =1:N
       Segment = Motion(:, 1 + (j-1)*Incr : L + (j-1)*Incr);
       Rms(i,j) = rms(Segment(i,:));
       Mav(i,j) = sum(abs(Segment(i,:)))/L;
       Ar(i, 1 + (j-1)*C : C*j ) = aryule(Segment(i,:),P); 
       
       for k = 1:(L-1)
            temp = abs(Segment(i,k+1) - Segment(i,k));
            Wl(i,j) = Wl(i,j) + temp ;
            if(abs(Segment(i, k) - Segment(i, k+1)) > T)
               Wamp(i,j) = Wamp(i,j)+1;
           end
        end
   end
end

for i = 1:N
    F1 = Mav(:,i)';
    F2 = Wamp(:,i)';
    F3 = Wl(:,i)';
    F4 = Ar(:,1 + (i-1)*C : (i*C))';
    F4 = F4(:)';
    F5 = Rms(:,i)';
    Feats(i, :) = [F1, F2, F3, F4, F5];
   
end




