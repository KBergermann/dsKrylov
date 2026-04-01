function [lamb, x, s_min, s_max] = dsRR(A, b, m, k, RSS_method, s)
% Deterministically sketched Rayleigh--Ritz (dsRR) for approximating a
% subset of the eigenvalues and -vectors or the matrix A [Alg. 5.3, 1]
% 
% Input:    A,              System matrix.
%           b,              Right-hand-side vector.
%           m,              Krylov subspace dimension.
%           k,              Truncation-parameter in k_truncated Arnoldi.
%           RSS_method,     (Over-)sampling method for row indices.
%           s,              Sketch size.
% 
% Output:   lamb,           Approximate eigenvalues of A.
%           phi,            Approximate eigenvectors of A.
%           s_min,          Approximate lower bound on ||A*x_i - lamb_i*x_i|| / ||S(A*x_i - lamb_i*x_i)||, for i=1,...,m.
%           s_max,          Approximate upper bound on ||A*x_i - lamb_i*x_i|| / ||S(A*x_i - lamb_i*x_i)||, for i=1,...,m.
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
    
    [VV, ~] = arnoldi_k_truncated(A,b,m,k); % line 1 in [Alg. 5.3, 1]
    
    switch RSS_method
        case 'deim'
            p = deim(VV); % line 2 in [Alg. 5.3, 1]
        case 'q_deim'
            p = q_deim(VV); % line 2 in [Alg. 5.3, 1]
        case 'gpode'
            p = gpode(VV, s); % over-sample, lines 3-5 in [Alg. 5.3, 1]
        case 'mpe'
            pp = deim(VV); % line 2 in [Alg. 5.3, 1]
            [p, ~] = fastMPE(VV, s, pp, 1); % over-sample, lines 3-5 in [Alg. 5.3, 1]
    end
    
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.3, 1]
    
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.3, 1]
    [YY, lamb] = eig(Rd \ (Qd' * (Sd * (A * VV(:,1:m))))); % solving the m-by-m eigenvalue problem, line 8 in [Alg. 5.3, 1]
    x = VV(:,1:m) * YY;
    x = x ./ vecnorm(x); % compute normalized eigenvector approximations, line 9 in [Alg. 5.3, 1]

    dsRR_sv_mpe = svd(VV(:,1:m) / Rd);
    s_min = min(dsRR_sv_mpe); % sigma_min(V_mR_m^{-1}) as lower bound
    s_max = max(dsRR_sv_mpe); % sigma_max(V_mR_m^{-1}) as upper bound
    
end
