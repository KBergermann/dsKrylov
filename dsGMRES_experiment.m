
% Code to reproduce the numerical experiments for approximately solving
% (I-A)x=b from [Sec. 6.2, 1] with deterministically sketched GMRES
% (dsGMRES) [Alg. 5.2, 1]
% 
% [1] K. Bergermann, Deterministic sketching for Krylov subspace methods,
%     Preprint (2026)
% [2] Y. Nakatsukasa and J. A. Tropp, Fast and accurate randomized
%     algorithms for linear systems and eigenvalue problems, SIAM Journal
%     on Matrix Analysis and Applications, 45 (2024), pp. 1183–1214.

addpath 'subroutines'

%% Parameters
m_max = 550; % maximum Krylov subspace dimension (multiple of 10)
k = 4; % truncation length in k-truncated Arnoldi [Alg. 2.1, 1]
tol = eps; % tolerance for gmres

%% Matrix (I-A)
d = 256; % Number of grid points per dimension

e = ones(d,1);
I = speye(d);
C = spdiags([e  -e], -1:0, d, d); % 1d convection operator
Aadv = kron(C,I) + kron(I,C'); % 2d convection operator
Adiff = - gallery('poisson',d,d); % 2d diffusion operator
A = (d-1)*Aadv + (1e-03*(d-1)^2)*Adiff; % convection-diffusion operator
t = 1;
A = speye(d^2,d^2) - t*A;
n = size(A,1);

%% Vector b
x = linspace(0,1,d); y = linspace(0,1,d); [X,Y] = meshgrid(x,y);
b = reshape(0.3*ones(d,d) + 256*(X.*Y.*(ones(d,d)-X).*(ones(d,d)-Y)).^2,[d^2, 1]);

%% generate non-orthogonal Krylov basis of maximum dimension
x0 = zeros(n,1); % starting guess
r = b - A*x0; % compute initial residual, line 1 in [Alg. 5.2, 1]
[VV, MM] = arnoldi_k_truncated(A, r, m_max, k); % line 2 in [Alg. 5.2, 1]
cond(VV)

%% Row subset selection with DEIM, line 3 in [Alg. 5.2, 1]
p_deim = deim(VV);

%% Loop over m

res_norm_GMRES = zeros(m_max,1);
res_norm_sGMRES = zeros(m_max,1);
res_norm_dsGMRES_deim = zeros(m_max,1);
res_norm_dsGMRES_mpe = zeros(m_max,1);
res_norm_dsGMRES_gpode = zeros(m_max,1);
cond_sGMRES = zeros(m_max,1);
cond_sGMRES_red = zeros(m_max,1);
cond_dsGMRES_deim = zeros(m_max,1);
cond_dsGMRES_deim_red = zeros(m_max,1);
cond_dsGMRES_mpe = zeros(m_max,1);
cond_dsGMRES_mpe_red = zeros(m_max,1);
cond_dsGMRES_gpode = zeros(m_max,1);
cond_dsGMRES_gpode_red = zeros(m_max,1);

m_range = sort([10:10:m_max, 512:2:518]);

bar=waitbar(0,'Looping over m...');
for m=m_range
    % sketch sizes
    s = 2*m;
    s_det1 = m+1;
    s_det2 = ceil(1.1*m);
    
    % orthogonal GMRES
    x_GMRES = gmres(A,b,m,tol,1); % plain GMRES without restarting
    r_GMRES = norm(b - A*x_GMRES);
    res_norm_GMRES(m) = r_GMRES;

    % sGMRES [2]
    hS = setup_dct_sketching_handle(n,s);
    g = hS(r);
    [QQ,RR] = qr(hS(MM(:,1:m)),0);
    x_sGMRES = x0 + VV(:,1:m) * (RR \ (QQ' * g));
    res_norm_sGMRES(m) = norm(b - A * x_sGMRES);
    cond_sGMRES(m) = cond(VV(:,1:m) / RR) * cond(hS(VV(:,1:m) / RR));
    cond_sGMRES_red(m) = cond(VV(:,1:m) / RR);
    

    % dsGMRES DEIM s=m
    p = p_deim(1:m);
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 7 in [Alg. 5.2, 1]
    g = Sd * r; % sketched residual
    [Qd,Rd] = qr(Sd * MM(:,1:m),0); % QR decomposition, line 8 in [Alg. 5.2, 1]
    x_dsGMRES_deim = x0 + VV(:,1:m) * (Rd \ (Qd' * g)); % dsGMRES approximation, lines 9+10 in [Alg. 5.2, 1]
    res_norm_dsGMRES_deim(m) = norm(b - A * x_dsGMRES_deim);
    cond_dsGMRES_deim(m) = cond(VV(:,1:m) / Rd) * cond(Sd * (VV(:,1:m) / Rd));
    cond_dsGMRES_deim_red(m) = cond(VV(:,1:m) / Rd);
    
    
    % dsGMRES MPE s=1.1m
    [p, ~] = fastMPE(VV(:,1:m), s_det2, p_deim(1:m), 1); % over-sample, lines 4-6 in [Alg. 5.2, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 7 in [Alg. 5.2, 1]
    g = Sd * r; % sketched residual
    [Qd,Rd] = qr(Sd * MM(:,1:m),0); % QR decomposition, line 8 in [Alg. 5.2, 1]
    x_dsGMRES_mpe = x0 + VV(:,1:m) * (Rd \ (Qd' * g)); % dsGMRES approximation, lines 9+10 in [Alg. 5.2, 1]
    res_norm_dsGMRES_mpe(m) = norm(b - A * x_dsGMRES_mpe);
    cond_dsGMRES_mpe(m) = cond(VV(:,1:m) / Rd) * cond(Sd * (VV(:,1:m) / Rd));
    cond_dsGMRES_mpe_red(m) = cond(VV(:,1:m) / Rd);
    
    
    % dsGMRES GPODE s=m+1
    p = gpode(VV(:,1:m), s_det1); % over-sample, lines 4-6 in [Alg. 5.2, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 7 in [Alg. 5.2, 1]
    g = Sd * r; % sketched residual
    [Qd,Rd] = qr(Sd * MM(:,1:m),0); % QR decomposition, line 8 in [Alg. 5.2, 1]
    x_dsGMRES_gpode = x0 + VV(:,1:m) * (Rd \ (Qd' * g)); % dsGMRES approximation, lines 9+10 in [Alg. 5.2, 1]
    res_norm_dsGMRES_gpode(m) = norm(b - A * x_dsGMRES_gpode);
    cond_dsGMRES_gpode(m) = cond(VV(:,1:m) / Rd)* cond(Sd * (VV(:,1:m) / Rd));
    cond_dsGMRES_gpode_red(m) = cond(VV(:,1:m) / Rd);
    
    
    waitbar(m/m_max,bar);
end
close(bar)

%% Plots [Fig. 3, 1]
figure(1)
semilogy(m_range,res_norm_GMRES(m_range),'k','Linewidth',2)
hold on
semilogy(m_range,res_norm_sGMRES(m_range),'g','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_deim(m_range),'b','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_mpe(m_range),'c','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_gpode(m_range),'r','Linewidth',2)
legend('GMRES','sGMRES, s=2m','dsGMRES, DEIM, s=m','dsGMRES, MPE, s=m+1','dsGMRES, GPODE, s=1.1m','Location','southwest')
xlabel('m')
ylabel('$\|(I-A)x_m^* - b\|$','interpreter','latex')
set(gca,'fontsize',14)
hold off


figure(2)
semilogy(m_range,ones(length(m_range),1),'k','Linewidth',2)
hold on
semilogy(m_range,res_norm_sGMRES(m_range)./res_norm_GMRES(m_range),'g','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_deim(m_range)./res_norm_GMRES(m_range),'b','Linewidth',2)
semilogy(m_range,cond_sGMRES(m_range),'g--','Linewidth',2)
semilogy(m_range,cond_sGMRES_red(m_range),'g:','Linewidth',2)
semilogy(m_range,cond_dsGMRES_deim(m_range),'b--','Linewidth',2)
semilogy(m_range,cond_dsGMRES_deim_red(m_range),'b:','Linewidth',2)
hold off
legend('GMRES','sGMRES, s=2m','dsGMRES, DEIM, s=m','sGMRES \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','sGMRES \kappa(V_mR_m^{-1})','dsGMRES DEIM \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','dsGMRES DEIM \kappa(V_mR_m^{-1})','Location','northwest')
ylim([1e-01, 1e11])
xlabel('m')
ylabel('$\frac{\|(I-A)x_m^* - b\|}{\|(I-A)x_m^{GMRES} - b\|}$','interpreter','latex')
set(gca,'fontsize',14)

figure(3)
semilogy(m_range,ones(length(m_range),1),'k','Linewidth',2)
hold on
semilogy(m_range,res_norm_sGMRES(m_range)./res_norm_GMRES(m_range),'g','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_gpode(m_range)./res_norm_GMRES(m_range),'r','Linewidth',2)
semilogy(m_range,cond_sGMRES(m_range),'g--','Linewidth',2)
semilogy(m_range,cond_sGMRES_red(m_range),'g:','Linewidth',2)
semilogy(m_range,cond_dsGMRES_gpode(m_range),'r--','Linewidth',2)
semilogy(m_range,cond_dsGMRES_gpode_red(m_range),'r:','Linewidth',2)
hold off
legend('GMRES','sGMRES, s=2m','dsGMRES, GPODE, s=m+1','sGMRES \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','sGMRES \kappa(V_mR_m^{-1})','dsGMRES GPODE \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','dsGMRES GPODE \kappa(V_mR_m^{-1})','Location','northwest')
xlabel('m')
ylabel('$\frac{\|(I-A)x_m^* - b\|}{\|(I-A)x_m^{GMRES} - b\|}$','interpreter','latex')
set(gca,'fontsize',14)

figure(4)
semilogy(m_range,ones(length(m_range),1),'k','Linewidth',2)
hold on
semilogy(m_range,res_norm_sGMRES(m_range)./res_norm_GMRES(m_range),'g','Linewidth',2)
semilogy(m_range,res_norm_dsGMRES_mpe(m_range)./res_norm_GMRES(m_range),'c','Linewidth',2)
semilogy(m_range,cond_sGMRES(m_range),'g--','Linewidth',2)
semilogy(m_range,cond_sGMRES_red(m_range),'g:','Linewidth',2)
semilogy(m_range,cond_dsGMRES_mpe(m_range),'c--','Linewidth',2)
semilogy(m_range,cond_dsGMRES_mpe_red(m_range),'c:','Linewidth',2)
hold off
legend('GMRES','sGMRES, s=2m','dsGMRES, MPE, s=1.1m','sGMRES \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','sGMRES \kappa(V_mR_m^{-1})','dsGMRES MPE \kappa(SV_mR_m^{-1})\kappa(V_mR_m^{-1})','dsGMRES MPE \kappa(V_mR_m^{-1})','Location','northwest')
xlabel('m')
ylabel('$\frac{\|(I-A)x_m^* - b\|}{\|(I-A)x_m^{GMRES} - b\|}$','interpreter','latex')
set(gca,'fontsize',14)
