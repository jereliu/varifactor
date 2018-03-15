# varifactor

> An inference toolbox for Bayesian Exponential-family Factor Analysis models. 

* Supports Gaussian/Poisson/Binomial-distributed observations, and Gaussian/Dirichlet-distributed factors
* Automated latent-dimension discovery.
* Six modern inference algorithms: 
    * MCMC: Metropolis-hasting, [Elliptical Slice Sampler](https://arxiv.org/abs/1001.0175), [NUTS](https://arxiv.org/abs/1111.4246)
    * Variational Methods: MFVI, [NFVI](https://arxiv.org/abs/1606.04934), [SVGD](https://arxiv.org/abs/1608.04471)
* Various convergence diagonostic measures:
    * Posterior Mean, Covariance, and 95% CI Coverage Probability
    * Posterior Log Likelihood
    * [Kernel Stein Discrepancy](https://arxiv.org/abs/1602.03253)

## to-do
* Feature:
    - [ ] Metric: log posterior 
    - [ ] Algorithm: NFVI with [IAF](https://gist.github.com/springcoil/4fda94fcde0934b04fc34967e0c952de)

* Refactor
    - [ ] Visualization module with option for run-time smoothing.
    - [ ] Unified metric module: (Metric -> MomentMetric/DensityMetric/DistMetric)
