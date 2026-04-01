
% Code to reproduce the numerical experiments for approximating (e^A)*b
% from [Sec. 6.1, 1] with deterministically sketched FOM (dsFOM) [Alg. 5.1, 1]
% 
% [1] K. Bergermann, Deterministic sketching for Krylov subspace methods,
%     Preprint (2026)
% [2] Y. Saad, Analysis of some Krylov subspace approximations to the
%     matrix exponential operator, SIAM Journal on Numerical Analysis, 29
%     (1992), pp. 209–228.
% [3] S. Güttel and M. Schweitzer, Randomized sketching for Krylov
%     approximations of large-scale matrix functions, SIAM Journal on
%     Matrix Analysis and Applications, 44 (2023), pp. 1073–1095.
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 

addpath 'subroutines'

%% Parameters
m_max = 280; % maximum Krylov subspace dimension (multiple of 10)
m_ref = 350; % Krylov subspace dimension for reference solution
k = 2; % truncation length in k-truncated Arnoldi [Alg. 2.1, 1]

%% Matrix A
d = 256; % Number of grid points per dimension

%%% Grid
hx = 2/(d-1); % mesh size
n = d^2; % degrees of freedom

%%% Initial conditions
x = linspace(-1,1,d); y = linspace(-1,1,d); [X,Y] = meshgrid(x,y);
u0 = 0.5*exp(-X.^2).*exp(-Y.^2);

%%% Laplacian
e = ones(d,1);
Dx = spdiags([e -2*e e], -1:1, d, d); % 1D finite difference matrix
% Apply Neumann boundary in 1D
Dx(1,1) = -1; Dx(1,2) = 1;
Dx(end,end) = -1; Dx(end,end-1) = 1;
I = speye(d);
D = 1/40; % diffusion constant
L = D/(hx^2)*(kron(I,Dx)+kron(Dx,I)); % 2d Neumann Laplacian

%%% Non-linearity
beta = 1/4;
g = @(u) beta*(u - u.^2);
Gn1 = g(u0(:));

%%% Augmented matrix
C = Gn1;
p = size(C,2);
Jp = spdiags([zeros(p,1), ones(p,1)], 0:1,p,p);
A = [L, C; sparse(p,n), Jp];
n = size(A,1);

%% Vector b
ep = zeros(p,1); ep(end) = 1;
b = [u0(:); ep];

%% Matrix function f
f = @(A) expm(A);

%% Reference solution
[V,H] = arnoldi(A, b, m_ref); % classical orthogonal Arnoldi
e1 = zeros(m_ref,1); e1(1) = 1;
fAb = norm(b) * V(:,1:m_ref) * f(H(1:m_ref,1:m_ref)) * e1;

%% Error estimate of reference solution [2]
hmp1m = H(m_ref+1,m_ref);
HH = [H(1:m_ref,1:m_ref), e1; zeros(1,m_ref+1)];
expHH = expm(HH);
err_est_fAb_ref = norm(b) * hmp1m *  expHH(end-1,end)

%% generate non-orthogonal Krylov basis of maximum dimension
[VV, ~] = arnoldi_k_truncated(A, b, m_max, k); % line 1 in [Alg. 5.1, 1]
cond_VV = cond(VV)

%% Row subset selection with DEIM, line 2 in [Alg. 5.1, 1]
p_deim = deim(VV);

%% Loop over Krylov subspace dimension (in steps of 10)

err_FOM = zeros(m_max,1);
err_sFOM = zeros(m_max,1);
err_dsFOM_deim = zeros(m_max,1);
err_dsFOM_mpe = zeros(m_max,1);
err_dsFOM_gpode = zeros(m_max,1);
cond_sFOM = zeros(m_max,1);
cond_dsFOM_deim = zeros(m_max,1);
cond_dsFOM_mpe = zeros(m_max,1);
cond_dsFOM_gpode = zeros(m_max,1);

bar=waitbar(0,'Looping over m...');
for m=10:10:m_max
    % sketch sizes
    s = 2*m;
    s_det1 = m+1;
    s_det2 = ceil(1.1*m);
    
    % orthogonal FOM
    e1 = zeros(m,1); e1(1) = 1;
    fAb_FOM = norm(b) * V(:,1:m) * f(H(1:m,1:m)) * e1;
    err_FOM(m) = norm(fAb_FOM - fAb);

    % sFOM [3]
    hS = setup_dct_sketching_handle(n,s);
    [QQ,RR] = qr(hS(VV(:,1:m)),0);
    fAb_sFOM = VV(:,1:m) * (RR \ (f(QQ' * hS(A * VV(:,1:m) / RR))) * (QQ' * (hS(b))));
    err_sFOM(m) = norm(fAb_sFOM - fAb);
    cond_sFOM(m) = cond(VV(:,1:m) / RR);
    
    % dsFOM DEIM s=m
    p = p_deim(1:m);
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.1, 1]
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.1, 1]
    fAb_dsFOM_deim = VV(:,1:m) * (Rd \ (f(Qd' * Sd * A * VV(:,1:m) / Rd) * (Qd' * (Sd * b)))); % dsFOM approximation, line 8 in [Alg. 5.1, 1]
    err_dsFOM_deim(m) = norm(fAb_dsFOM_deim - fAb);
    cond_dsFOM_deim(m) = cond(VV(:,1:m) / Rd);
    
    % dsFOM MPE s=1.1m
    [p, ~] = fastMPE(VV(:,1:m), s_det2, p_deim(1:m), 1); % over-sample, lines 3-5 in [Alg. 5.1, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.1, 1]
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.1, 1]
    fAb_dsFOM_mpe = VV(:,1:m) * (Rd \ (f(Qd' * Sd * A * VV(:,1:m) / Rd) * (Qd' * (Sd * b)))); % dsFOM approximation, line 8 in [Alg. 5.1, 1]
    err_dsFOM_mpe(m) = norm(fAb_dsFOM_mpe - fAb);
    cond_dsFOM_mpe(m) = cond(VV(:,1:m) / Rd);
    
    % dsFOM GappyPOD+E s=m+1
    p = gpode(VV(:,1:m), s_det1); % over-sample, lines 3-5 in [Alg. 5.1, 1]
    I = speye(n);
    Sd = I(p,:); % set-up deterministic sketching matrix, line 6 in [Alg. 5.1, 1]
    [Qd,Rd] = qr(Sd * VV(:,1:m),0); % basis whitening, line 7 in [Alg. 5.1, 1]
    fAb_dsFOM_gpode = VV(:,1:m) * (Rd \ (f(Qd' * Sd * A * VV(:,1:m) / Rd) * (Qd' * (Sd * b)))); % dsFOM approximation, line 8 in [Alg. 5.1, 1]
    err_dsFOM_gpode(m) = norm(fAb_dsFOM_gpode - fAb);
    cond_dsFOM_gpode(m) = cond(VV(:,1:m) / Rd);
    
    waitbar(m/m_max,bar);
end
close(bar)

%% Plots [Fig. 2, 1]
figure(1)
semilogy(10:10:m_max,err_FOM(10:10:m_max),'k','Linewidth',2)
hold on
semilogy(10:10:m_max,err_sFOM(10:10:m_max),'g','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_deim(10:10:m_max),'b','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_mpe(10:10:m_max),'c','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_gpode(10:10:m_max),'r','Linewidth',2)
legend('FOM','sFOM, s=2m','dsFOM, DEIM, s=m','dsFOM, MPE, s=m+1','dsFOM, GPODE, s=1.1m','Location','southwest')
xlabel('m')
ylabel('$\|f_m^* - f(A)b\|$','interpreter','latex')
set(gca,'fontsize',14)
hold off

figure(2)
semilogy(10:10:m_max,ones(m_max/10,1),'k','Linewidth',2)
hold on
semilogy(10:10:m_max,err_sFOM(10:10:m_max)./err_FOM(10:10:m_max),'g','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_deim(10:10:m_max)./err_FOM(10:10:m_max),'b','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_mpe(10:10:m_max)./err_FOM(10:10:m_max),'c','Linewidth',2)
semilogy(10:10:m_max,err_dsFOM_gpode(10:10:m_max)./err_FOM(10:10:m_max),'r','Linewidth',2)
semilogy(10:10:m_max,cond_sFOM(10:10:m_max),'g--','Linewidth',2)
semilogy(10:10:m_max,cond_dsFOM_deim(10:10:m_max),'b--','Linewidth',2)
semilogy(10:10:m_max,cond_dsFOM_mpe(10:10:m_max),'c--','Linewidth',2)
semilogy(10:10:m_max,cond_dsFOM_gpode(10:10:m_max),'r--','Linewidth',2)
hold off
legend('FOM','sFOM, s=2m','dsFOM, DEIM, s=m','dsFOM, MPE, s=m+1','dsFOM, GPODE, s=1.1m','Location','northwest')
xlabel('m')
ylabel('$\|f_m^* - f(A)b\| / \|f_m^{FOM} - f(A)b\|$','interpreter','latex')
set(gca,'fontsize',14)
