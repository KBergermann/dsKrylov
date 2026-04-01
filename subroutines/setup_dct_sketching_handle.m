function hS = setup_dct_sketching_handle(N,s)
% Set-up of a function handle for multiplication with the SRFT using DCT
% Taken from:
% https://github.com/marcelschweitzer/sketched_fAb
rng(0)
E = spdiags(2*round(rand(N,1))-1,0,N,N); % Rademacher
D = speye(N); D = D(randperm(N,s),:);
hS = @(X) D*dct(E*X)/sqrt(s/N);
