function [V,H] = arnoldi(A, b, m)
% Implementation of the classical Arnoldi method to construct an
% orthonormal basis for the Krylov subspace K_m(A,b).
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 
    n = size(A,1);
    V = zeros(n,m+1);
    H = zeros(m+1,m);
    V(:,1) = b/norm(b);
    for i=1:m
        w = A * V(:,i);
        for j=1:i
            H(j,i) = V(:,j)'*w;
            w = w - H(j,i)*V(:,j);
        end
        H(i+1,i) = norm(w);
        V(:,i+1) = w/norm(H(i+1,i));
    end
end
