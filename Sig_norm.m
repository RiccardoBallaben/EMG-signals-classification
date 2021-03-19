function [sig_norm] = Sig_norm(sig)
%Normalizes the signal values using the peak value

[m, I] = max(sig);
sig_norm = sig./sig(I);

end

