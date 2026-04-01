function p = deim(V)
% Discrete empirical interpolation method (DEIM) extracting k row indices
% from the basis V into the index vector p.
% Implementation uses updates of the LU decomposition for the linear system solve.
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 
    k=size(V,2);
    [~,p(1)] = max(abs(V(:,1)));
    % setup LU decomposition
    L = 1;
    U = V(p,1);
    for j=2:k
        c = U\(L\V(p,j));
        rk = V(:,j)-V(:,1:j-1)*c;
        [~,p(j)]  = max(abs(rk));
        % update LU decomposition
        b = V(p(1:j-1),j);
        c = V(p(j),1:j-1)';
        d = V(p(j),j);
        x = (U'\c);
        u = L\b;
        L = [L, zeros(j-1,1); x', 1];
        U = [U, u; zeros(1,j-1), d-x'*u];
    end
end
