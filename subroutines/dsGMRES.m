function [x, r, err] = dsGMRES(A, b, m, x0, k, RSS_method, s)
% Deterministically sketched generalized minimal residual method (dsGMRES)
% [Alg. 5.2, 1]
% 
% Input:    A,          System matrix.
%           b,          Right-hand-side vector.
%           m,          Krylov subspace dimension.
%           k,          Truncation-parameter in k_truncated Arnoldi.
%           RSS_method, (Over-)sampling method for row indices.
%           s,          Sketch size.
% 
% Output:   x,          dsGMRES approximation to Ax=b.
%           r,          dsGMRES residual norm ||Ax_dsGMRES-b||
%           err,        (non-sharp) error bound on ||Ax_dsGMRES-b|| / ||Ax_GMRES-b||
% 
% [1] K. Bergermann, Deterministic sketching for Krylov subspace methods,
%     Preprint (2026)
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 

    n = size(A,1);
    
    if nargin < 4
        x0 = zeros(n,1);
        if nargin < 5
            k = 4;
            if nargin < 6
                RSS_method = 'mpe';
                if nargin < 7
                    s = ceil(1.1*m);
                end
            end
        end
    end
    
    r = b - A * x0; % compute initial residual, line 1 in [Alg. 5.2, 1]

    [VV, MM] = arnoldi_k_truncated(A,r,m,k); % line 2 in [Alg. 5.2, 1]
    
    switch RSS_method
        case 'deim'
            p = deim(VV); % line 3 in [Alg. 5.2, 1]
        case 'q_deim'
            p = q_deim(VV); % line 3 in [Alg. 5.2, 1]
        case 'gpode'
            p = gpode(VV, s); % over-sample, lines 4-6 in [Alg. 5.2, 1]
        case 'mpe'
            pp = deim(VV); % line 3 in [Alg. 5.2, 1]
            [p, ~] = fastMPE(VV, s, pp, 1); % over-sample, lines 4-6 in [Alg. 5.2, 1]
    end
    
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 7 in [Alg. 5.2, 1]
    
    g = Sd * r; % sketched residual
    [Qd,Rd] = qr(Sd * MM,0); % QR decomposition, line 8 in [Alg. 5.2, 1]
    x = x0 + VV * (Rd \ (Qd' * g)); % dsGMRES approximation, lines 9+10 in [Alg. 5.2, 1]
    r = norm(b - A * x);
    err = cond(Sd * (VV(:,1:m) / Rd)) * cond(VV(:,1:m) / Rd); % subspace embedding distortion factor [Eq. (3.3), 1]

end
