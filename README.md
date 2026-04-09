# dsKrylov
Deterministic sketching for Krylov subspace methods

This repository accompanies the

**Paper**

[1] K. Bergermann, Deterministic sketching for Krylov subspace methods, Preprint, [arXiv:2604.07158](https://arxiv.org/abs/2604.07158) (2026).

**Scripts**

There are three scripts reproducing the numerical experiments from [1] as well as a demo file that illustrates all examples on small problems that are fast to execute and ready to play with:

- dsFOM_experiment.m, reproduces the numerical experiments for approximating (e^A)*b from [Sec. 6.1, 1] with deterministically sketched FOM (dsFOM) [Alg. 5.1, 1],
- dsGMRES_experiment.m, reproduces the numerical experiments for approximately solving (I-A)x=b from [Sec. 6.2, 1] with deterministically sketched GMRES (dsGMRES) [Alg. 5.2, 1],
- dsRR_experiment.m, reproduce the numerical experiments for approximating a subset of eigenvalues and -vectors of the graph Laplacian L from [Sec. 6.3, 1] with deterministically sketched Rayleigh--Ritz (dsRR) [Alg. 5.3, 1].
- demo.m, deterministically sketched Krylov methods playground.

**Directory data**

contains two directed networks for the dsRR experiments

- gre_1107.mat, small network with 1,107 nodes. Taken from https://sparse.tamu.edu/HB/gre_1107,
- web-Stanford.mat, bigger network with 281,903 nodes. Taken from https://sparse.tamu.edu/SNAP/web-Stanford.

**Directory subroutines**

- arnoldi.m, implementation of the classical Arnoldi method to construct an orthonormal basis for the Krylov subspace K_m(A,b),
- arnoldi_k_truncated.m, implementation of the k-truncated Arnoldi method to construct a non-orthogonal basis for the Krylov subspace K_m(A,b),
- deim.m, Discrete empirical interpolation method (DEIM) extracting k row indices from the basis V into the index vector p. Implementation uses updates of the LU decomposition for the linear system solve.
- dsFOM.m, implementation of deterministically sketched FOM (dsFOM) for approximating f(A)b [Alg. 5.1, 1],
- dsGMRES.m, implementation of the deterministically sketched generalized minimal residual method (dsGMRES) [Alg. 5.2, 1],
- dsRR.m, implementation of deterministically sketched Rayleigh--Ritz (dsRR) for approximating a subset of the eigenvalues and -vectors or the matrix A [Alg. 5.3, 1],
- fastMPE.m, Accelerated MPE point selection implementing the method from [2]. Taken from: https://github.com/RalfZimmermannSDU/RandomizedGreedyMPS,
- gpode.m, GappyPOD+E code that extracts m row indices from the basis U into the index vector p. Taken from [Alg. 1, 3],
- q_deim.m, implementation of Q-DEIM for extracting row indices from a basis. Taken from [4],
- setup_dct_sketching_handle.m, Set-up of a function handle for multiplication with the SRFT using DCT. Taken from the repository accompanying [5]: https://github.com/marcelschweitzer/sketched_fAb,
- setup_example.m, Sets up examples for the deterministic Krylov method playground (demo.m),
- setup_fwht_sketching_handle.m, Set-up of a function handle for multiplication with the SRFT using FWHT. Adapted from the DCT version from the repository accompanying [5]: https://github.com/marcelschweitzer/sketched_fAb.

**Versions**

All codes were tested with Matlab 2020b on Ubuntu 20.04.6 LTS and Matlab 2025a on Ubuntu 24.04.4 LTS.

**References**

- [1] K. Bergermann, Deterministic sketching for Krylov subspace methods, Preprint (2026).
- [2] R. Zimmermann and K. Willcox, An accelerated greedy missing point estimation procedure, SIAM Journal on Scientific Computing, 38 (2016), pp. A2827–A2850.
- [3] B. Peherstorfer, Z. Drmac, and S. Gugercin, Stability of discrete empirical interpolation and gappy proper orthogonal decomposition with randomized and deterministic sampling points, SIAM Journal on Scientific Computing, 42 (2020), pp. A2837–A2864.
- [4] Z. Drmac and S. Gugercin, A new selection operator for the discrete empirical interpolation method—improved a priori error bound and extensions, SIAM Journal on Scientific Computing, 38 (2016), pp. A631–A648.
- [5] S. Güttel and M. Schweitzer, Randomized sketching for krylov approximations of large-scale matrix functions, SIAM Journal on Matrix Analysis and Applications, 44 (2023), pp. 1073–1095.

**Author**

Kai Bergermann (kai.bergermann@sns.it)

