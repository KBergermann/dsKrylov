function [V, M] = arnoldi_k_truncated(A, b, m, k)
% Implementation of the k-truncated Arnoldi method to construct a
% non-orthogonal basis for the Krylov subspace K_m(A,b).
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 
    n = size(A,1);
    V = zeros(n,m);
    M = zeros(n,m);
    V(:,1) = b/norm(b);
    M(:,1) = A * V(:,1);
    for j=2:m
        w = M(:,j-1);
        Vw = V(:,max(j-k,1):j-1)'*w;
        w = w - V(:,max(j-k,1):j-1)*Vw;
        V(:,j) = w/norm(w);
        M(:,j) = A * V(:,j);
    end
end
