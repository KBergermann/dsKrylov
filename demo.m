
%%% Deterministically sketched Krylov methods playground %%%
% Manipulate subroutines/setup_example.m for the full experience
% 
% Kai Bergermann, 2026
% https://github.com/KBergermann/dsKrylov
% 

addpath('subroutines')

%%% dsFOM %%%

d = 32;
m = 35;
args = setup_example('dsFOM', d);
[fAb, ~] = dsFOM(args.A, args.b, args.f, m);
dsFOM_err = norm(fAb - args.f(args.A)*args.b)
figure(1)
subplot(121)
imagesc(reshape(args.b(1:end-1),[d,d]))
colorbar
title('Before')
subplot(122)
imagesc(reshape(fAb(1:end-1),[d,d]))
colorbar
title('After')
sgtitle('One time step of the exponential Euler of a reaction-diffusion example')

%%% dsGMRES %%%

d = 32;
m = 70;
args = setup_example('dsGMRES', d);
[x, r, ~] = dsGMRES(args.A, args.b, m);
dsGMRES_res_norm = r
figure(2)
subplot(121)
imagesc(reshape(args.b,[d,d]))
colorbar
title('Before')
subplot(122)
imagesc(reshape(x,[d,d]))
colorbar
title('After')
sgtitle('One time step of the implicit Euler of an advection-diffusion example')

%%% dsRR %%%

m = 150;
args = setup_example('dsRR');
[lamb, x, ~, ~] = dsRR(args.A, args.b, m);
[~,ind] = sort(abs(diag(lamb)));
dsRR_Fiedler_res_norm = norm(args.A*x(:,ind(2)) - lamb(ind(2),ind(2))*x(:,ind(2)))
adj = diag(diag(args.A)) - args.A;
G = digraph(adj);
G.Nodes.NodeColors = sign(real(x(:,ind(2))));
[~,perm] = sort(real(x(:,ind(2))));
figure(3)
subplot(121)
plot(G,'NodeCData',G.Nodes.NodeColors)
title('Fiedler vector signs')
subplot(122)
spy(adj(perm,perm))
title('Adjacency permutation')
sgtitle('Graph Laplacian Fiedler vector example')
