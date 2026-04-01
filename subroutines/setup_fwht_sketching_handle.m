function hS = setup_fwht_sketching_handle(N,s)
% Set-up of a function handle for multiplication with the SRFT using FWHT
% Adapted from:
% https://github.com/marcelschweitzer/sketched_fAb
rng(0)
NN = pow2(ceil(log2(N))); % next power of 2
E = [spdiags(2*round(rand(N,1))-1,0,N,N); sparse(NN-N,N)]; % Rademacher
D = speye(NN); D = D(randperm(NN,s),:);
hS = @(X) D*fwht(E*X)/sqrt(s/N);
