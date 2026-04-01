function argout = setup_example(example, d)
% Sets up examples for the deterministic Krylov method playground (demo.m)
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 

if nargin < 2
    d = 32;
end

argout = struct();

switch example
    case 'dsFOM'
        % Matrix A
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

        % Vector b
        ep = zeros(p,1); ep(end) = 1;
        b = [u0(:); ep];

        % Matrix function f
        f = @(A) expm(A);
        
        argout.A = A;
        argout.b = b;
        argout.f = f;
        
    case 'dsGMRES'
        % Matrix (I-A)
        e = ones(d,1);
        I = speye(d);
        C = spdiags([e  -e], -1:0, d, d); % 1d convection operator
        Aadv = kron(C,I) + kron(I,C'); % 2d convection operator
        Adiff = - gallery('poisson',d,d); % 2d diffusion operator
        A = (d-1)*Aadv + (1e-03*(d-1)^2)*Adiff; % convection-diffusion operator
        t = 1;
        A = speye(d^2,d^2) - t*A;

        % Vector b
        x = linspace(0,1,d); y = linspace(0,1,d); [X,Y] = meshgrid(x,y);
        b = reshape(0.3*ones(d,d) + 256*(X.*Y.*(ones(d,d)-X).*(ones(d,d)-Y)).^2,[d^2, 1]);

        argout.A = A;
        argout.b = b;
        
    case 'dsRR'
        % Matrix A
        load data/gre_1107.mat Problem
        A = Problem.A > 0;
        n = size(A,1);
        A = A - spdiags(A,0,n,n);
        d_in = sum(A,1)'; % vector of node in-degrees
        while sum(d_in==0) > 0 % recursively deleting isolated in-degree nodes, since removal of one node may isolate another
            ind = (d_in ~= 0);
            A = A(ind,ind);
            n = size(A,1);
            d_in = sum(A,1)';
        end
        A = speye(n,n) - spdiags(d_in.^(-1/2),0,n,n) * A * spdiags(d_in.^(-1/2),0,n,n); % normalized graph Laplacian
        n = size(A,1);

        % Vector b
        rng(0)
        b = rand(n,1);
        
        argout.A = A;
        argout.b = b;
end