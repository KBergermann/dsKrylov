function [fAb, err] = dsFOM(A, b, f, m, k, RSS_method, s)
% Deterministically sketched FOM (dsFOM) for approximating f(A)b
% [Alg. 5.1, 1]
% 
% Input:    A,              System matrix.
%           b,              Right-hand-side vector.
%           f,              Matrix function (handle).
%           m,              Krylov subspace dimension.
%           k,              Truncation-parameter in k_truncated Arnoldi.
%           RSS_method,     (Over-)sampling method for row indices.
%           s,              Sketch size.
% 
% Output:   fAb,            dsGMRES approximation to Ax=b.
%           err,            condition number suspected to be an error bound on ||fAb_dsFOM-f(A)b|| / ||fAb_FOM-f(A)b||
% 
% [1] K. Bergermann, Deterministic sketching for Krylov subspace methods,
%     Preprint (2026)
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 
    n = size(A,1);
    
    if nargin < 5
        k = 4;
        if nargin < 6
            RSS_method = 'mpe';
            if nargin < 7
                s = ceil(1.1*m);
            end
        end
    end
    
    [VV, ~] = arnoldi_k_truncated(A,b,m,k); % line 1 in [Alg. 5.1, 1]
    
    switch RSS_method
        case 'deim'
            p = deim(VV); % line 2 in [Alg. 5.1, 1]
        case 'q_deim'
            p = q_deim(VV); % line 2 in [Alg. 5.1, 1]
        case 'gpode'
            p = gpode(VV, s); % over-sample, lines 3-5 in [Alg. 5.1, 1]
        case 'mpe'
            pp = deim(VV); % line 2 in [Alg. 5.1, 1]
            [p, ~] = fastMPE(VV, s, pp, 1); % over-sample, lines 3-5 in [Alg. 5.1, 1]
    end
    
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.1, 1]
    
    [Qd,Rd] = qr(Sd * VV,0); % basis whitening, line 7 in [Alg. 5.1, 1]
    fAb = VV * (Rd \ (f(Qd' * Sd * A * VV(:,1:m) / Rd) * (Qd' * (Sd * b)))); % dsFOM approximation, line 8 in [Alg. 5.1, 1]
    err = cond(VV(:,1:m) / Rd); % subspace embedding distortion factor [Cor. 3.2, 1]
    
end