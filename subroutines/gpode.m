function p = gpode (U, m)
% GappyPOD+E code that extracts m row indices from the basis U
% into the index vector p.
% 
% Taken from Algorithm 1 of:
% B. Peherstorfer, Z. Drmac, and S. Gugercin, Stability of discrete
% empirical interpolation and gappy proper orthogonal decomposition
% with randomized and deterministic sampling points, SIAM Journal
% on Scientific Computing, 42 (2020), pp. A2837–A2864.
    [~, ~, p] = qr (U', 'vector');
    p = p(1:size(U,2))';
    for i= length(p)+1:m
        [~, S, W] = svd(U(p,:), 0);
        g = S(end-1, end-1).^2 - S(end, end)^2;
        Ub = W'*U';
        r = g + sum(Ub.^2, 1);
        r = r - sqrt((g + sum(Ub.^2,1)).^2 -4*g*Ub(end,:).^2);
        [~, I] = sort(r,'descend');
        e = 1;
        while any(I(e) == p)
            e = e + 1;
        end
        p(end + 1) = I(e);
    end
end
