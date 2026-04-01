
% Code to reproduce the numerical experiments for approximating a subset
% of eigenvalues and -vectors of the graph Laplacian L from [Sec. 6.3, 1] 
% with deterministically sketched Rayleigh--Ritz (dsRR) [Alg. 5.3, 1]
% 
% [1] K. Bergermann, Deterministic sketching for Krylov subspace methods,
%     Preprint (2026)
% [2] Y. Nakatsukasa and J. A. Tropp, Fast and accurate randomized
%     algorithms for linear systems and eigenvalue problems, SIAM Journal
%     on Matrix Analysis and Applications, 45 (2024), pp. 1183–1214.

addpath('subroutines')

%% Parameters
m_max = 150; % maximum Krylov subspace dimension (multiple of 10)
k = 8; % truncation length in k-truncated Arnoldi [Alg. 2.1, 1]

%% Matrix L
load data/web-Stanford.mat
A = Problem.A;
d_in = sum(A,1)'; % vector of node in-degrees
while sum(d_in==0) > 0 % recursively deleting isolated in-degree nodes, since removal of one node may isolate another
    ind = (d_in ~= 0);
    A = A(ind,ind);
    n = size(A,1);
    d_in = sum(A,1)';
end
L = speye(n,n) - spdiags(d_in.^(-1/2),0,n,n) * A * spdiags(d_in.^(-1/2),0,n,n); % normalized graph Laplacian
n = size(L,1);

%% Vector b
rng(0)
b = rand(n,1);

%% Generate Krylov bases of maximum dimension
[V,H] = arnoldi(L, b, m_max); % classical orthogonal Arnoldi
[VV, MM] = arnoldi_k_truncated(L, b, m_max, k); % line 1 in [Alg. 5.3, 1]
cond(VV)

%% Row subset selection with DEIM, line 2 in [Alg. 5.3, 1]
p_deim = deim(VV);

%% Loop over m

RR_Fiedler_residual = zeros(m_max,1);
sRR_Fiedler_residual = zeros(m_max,1);
dsRR_Fiedler_residual_gpode = zeros(m_max,1);
dsRR_Fiedler_residual_mpe = zeros(m_max,1);
sRR_Fiedler_residual_sketched = zeros(m_max,1);
dsRR_Fiedler_residual_sketched_gpode = zeros(m_max,1);
dsRR_Fiedler_residual_sketched_mpe = zeros(m_max,1);
sRR_sigma_min = zeros(m_max,1);
sRR_sigma_max = zeros(m_max,1);
dsRR_sigma_min_mpe = zeros(m_max,1);
dsRR_sigma_max_mpe = zeros(m_max,1);
dsRR_sigma_min_gpode = zeros(m_max,1);
dsRR_sigma_max_gpode = zeros(m_max,1);

bar=waitbar(0,'Looping over m...');
for m=10:10:m_max
    % sketch sizes
    s = 4*m;
    s_det = ceil(1.5*m);

    % orthogonal Rayleigh--Ritz
    [U, lamb_RR] = eig(H(1:m,1:m));
    phi_RR = V(:,1:m) * U;
    [~,I_lamb_RR] = sort(abs(diag(lamb_RR))); % sort eigenvalues by magnitude to identify Fiedler vector approximation
    RR_Fiedler_residual(m) = norm(L * phi_RR(:,I_lamb_RR(2)) - lamb_RR(I_lamb_RR(2),I_lamb_RR(2)) * phi_RR(:,I_lamb_RR(2)));
    

    % sRR [2]
    hS = setup_dct_sketching_handle(n,s);
    [QQ,RR] = qr(hS(VV(:,1:m)),0);
    [YY, lamb_sRR] = eig(RR \ (QQ' * (hS(L * VV(:,1:m)))));
    phi_sRR = VV(:,1:m) * YY;
    phi_sRR = phi_sRR ./ vecnorm(phi_sRR);
    [~,I_lamb_sRR] = sort(abs(diag(lamb_sRR))); % sort eigenvalues by magnitude to identify Fiedler vector approximation
    sRR_Fiedler_residual(m) = norm(L * phi_sRR(:,I_lamb_sRR(2)) - lamb_sRR(I_lamb_sRR(2),I_lamb_sRR(2)) * phi_sRR(:,I_lamb_sRR(2)));
    sRR_Fiedler_residual_sketched(m) = norm(hS(L * phi_sRR(:,I_lamb_sRR(2)) - lamb_sRR(I_lamb_sRR(2),I_lamb_sRR(2)) * phi_sRR(:,I_lamb_sRR(2))));
    sRR_sv = svd(VV(:,1:m) / RR);
    sRR_sigma_min(m) = min(sRR_sv); % sigma_min(V_m R_m^{-1}) as lower bound
    sRR_sigma_max(m) = max(sRR_sv); % sigma_max(V_m R_m^{-1}) as upper bound

    
    % dsRR MPE s=1.5m
    [p, ~] = fastMPE(VV(:,1:m), s_det, p_deim(1:m), 1); % over-sample, lines 3-5 in [Alg. 5.3, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.3, 1]
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.3, 1]
    [YY, lamb_dsRR_mpe] = eig(Rd \ (Qd' * (Sd * (L * VV(:,1:m))))); % solving the m-by-m eigenvalue problem, line 8 in [Alg. 5.3, 1]
    phi_dsRR_mpe = VV(:,1:m) * YY;
    phi_dsRR_mpe = phi_dsRR_mpe ./ vecnorm(phi_dsRR_mpe); % compute normalized eigenvector approximations, line 9 in [Alg. 5.3, 1]
    [~,I_lamb_dsRR] = sort(abs(diag(lamb_dsRR_mpe))); % sort eigenvalues by magnitude to identify Fiedler vector approximation
    dsRR_Fiedler_residual_mpe(m) = norm(L * phi_dsRR_mpe(:,I_lamb_dsRR(2)) - lamb_dsRR_mpe(I_lamb_dsRR(2),I_lamb_dsRR(2)) * phi_dsRR_mpe(:,I_lamb_dsRR(2)));
    dsRR_Fiedler_residual_sketched_mpe(m) = norm(Sd * (L * phi_dsRR_mpe(:,I_lamb_dsRR(2)) - lamb_dsRR_mpe(I_lamb_dsRR(2),I_lamb_dsRR(2)) * phi_dsRR_mpe(:,I_lamb_dsRR(2))));
    dsRR_sv_mpe = svd(VV(:,1:m) / Rd);
    dsRR_sigma_min_mpe(m) = min(dsRR_sv_mpe); % sigma_min(V_m R_m^{-1}) as lower bound
    dsRR_sigma_max_mpe(m) = max(dsRR_sv_mpe); % sigma_max(V_m R_m^{-1}) as upper bound

    
    % dsRR GPODE s=1.5m
    p = gpode(VV(:,1:m), s_det); % over-sample, lines 3-5 in [Alg. 5.3, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.3, 1]
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.3, 1]
    [YY, lamb_dsRR_gpode] = eig(Rd \ (Qd' * (Sd * (L * VV(:,1:m))))); % solving the m-by-m eigenvalue problem, line 8 in [Alg. 5.3, 1]
    phi_dsRR_gpode = VV(:,1:m) * YY;
    phi_dsRR_gpode = phi_dsRR_gpode ./ vecnorm(phi_dsRR_gpode); % compute normalized eigenvector approximations, line 9 in [Alg. 5.3, 1]
    [~,I_lamb_dsRR] = sort(abs(diag(lamb_dsRR_gpode))); % sort eigenvalues by magnitude to identify Fiedler vector approximation
    dsRR_Fiedler_residual_gpode(m) = norm(L * phi_dsRR_gpode(:,I_lamb_dsRR(2)) - lamb_dsRR_gpode(I_lamb_dsRR(2),I_lamb_dsRR(2)) * phi_dsRR_gpode(:,I_lamb_dsRR(2)));
    dsRR_Fiedler_residual_sketched_gpode(m) = norm(Sd * (L * phi_dsRR_gpode(:,I_lamb_dsRR(2)) - lamb_dsRR_gpode(I_lamb_dsRR(2),I_lamb_dsRR(2)) * phi_dsRR_gpode(:,I_lamb_dsRR(2))));
    dsRR_sv_gpode = svd(VV(:,1:m) / Rd);
    dsRR_sigma_min_gpode(m) = min(dsRR_sv_gpode); % sigma_min(V_m R_m^{-1}) as lower bound
    dsRR_sigma_max_gpode(m) = max(dsRR_sv_gpode); % sigma_max(V_m R_m^{-1}) as upper bound

    waitbar(m/m_max,bar);
end
close(bar)

%% Plots [Fig. 4, 1]
figure(1)
scatter(diag(abs(lamb_RR)),vecnorm(L * phi_RR - phi_RR * lamb_RR),30,'kx','Linewidth',2)
hold on
scatter(diag(abs(lamb_sRR)),vecnorm(L * phi_sRR - phi_sRR * lamb_sRR),30,'gx','Linewidth',2)
scatter(diag(abs(lamb_dsRR_gpode)),vecnorm(L * phi_dsRR_gpode - phi_dsRR_gpode * lamb_dsRR_gpode),30,'rx','Linewidth',2)
set(gca,'yscale','log')
legend('RR','sRR s=4m','dsRR GPODE s=1.5m','Location','southeast')
xlabel('$|\lambda_i|$','interpreter','latex')
ylabel('$\|Lx_i^*-\lambda_i^*x_i^*\|$','interpreter','latex')
hold off
set(gca,'fontsize',14)

figure(2)
semilogy(10:10:m_max, RR_Fiedler_residual(10:10:m_max),'k','Linewidth',2)
hold on
semilogy(10:10:m_max, sRR_Fiedler_residual(10:10:m_max),'g','Linewidth',2)
semilogy(10:10:m_max, dsRR_Fiedler_residual_mpe(10:10:m_max),'c','Linewidth',2)
semilogy(10:10:m_max, dsRR_Fiedler_residual_gpode(10:10:m_max),'r','Linewidth',2)
legend('RR','sRR s=4m','dsRR MPE s=1.5m','dsRR GPODE s=1.5m')
xlabel('m')
ylabel('$\|Lx_2^*-\lambda_2^*x_2^*\|$','interpreter','latex')
hold off
set(gca,'fontsize',14)

figure(3)
semilogy(10:10:m_max, sRR_Fiedler_residual(10:10:m_max) ./ sRR_Fiedler_residual_sketched(10:10:m_max),'g','Linewidth',2)
hold on
semilogy(10:10:m_max, dsRR_Fiedler_residual_mpe(10:10:m_max) ./ dsRR_Fiedler_residual_sketched_mpe(10:10:m_max),'c','Linewidth',2)
semilogy(10:10:m_max, dsRR_Fiedler_residual_gpode(10:10:m_max) ./ dsRR_Fiedler_residual_sketched_gpode(10:10:m_max),'r','Linewidth',2)
semilogy(10:10:m_max, sRR_sigma_min(10:10:m_max),'g--','Linewidth',2)
semilogy(10:10:m_max, sRR_sigma_max(10:10:m_max),'g:','Linewidth',2)
semilogy(10:10:m_max, dsRR_sigma_min_mpe(10:10:m_max),'c--','Linewidth',2)
semilogy(10:10:m_max, dsRR_sigma_max_mpe(10:10:m_max),'c:','Linewidth',2)
semilogy(10:10:m_max, dsRR_sigma_min_gpode(10:10:m_max),'r--','Linewidth',2)
semilogy(10:10:m_max, dsRR_sigma_max_gpode(10:10:m_max),'r:','Linewidth',2)
legend('sRR s=4m','dsRR MPE s=1.5m','dsRR GPODE s=1.5m')
xlabel('m')
ylabel('$\frac{\|Lx_2^*-\lambda_2^*x_2^*\|}{\|S(Lx_2^*-\lambda_2^*x_2^*)\|}$','interpreter','latex')
hold off
set(gca,'fontsize',14)
