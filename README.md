# Riemann Manifold HMCMC

Implementation of Riemann Manifold Langevin and Hamiltonian Monte Carlo of Girolami and al.

## Installation

`pip install tqdm`


## Code structure

- riemann_hmcmc.py that contains the mass class used for sampling
- random_vector.py that contains classes that help update theta and p according to their numerical schemes
- metric_tensor.py regroups the different metric tensors, associated to Riemann Manifold metrics that correspond
to G_i