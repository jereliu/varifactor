# varifactor

> An inference toolbox for Bayesian Exponential-family latent factor models. 

* Flexible distribution support:
    * Observation: Gaussian/Poisson/Binomial (lognormal is also possible)
    * Latent factor: Gaussian/Dirichlet
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
    - [ ] **Metric**: log posterior 
    - [ ] **Kernel**: IMQ Kernel 
    - [ ] **Metric**: [Maximum-Mean Discrepancy](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
    - [ ] **Algorithm**: NFVI with [IAF](https://gist.github.com/springcoil/4fda94fcde0934b04fc34967e0c952de)
    - [ ] **Example**: inference with lognormal outcome

* Refactor
    - [x] Redefine Model so U, V ~ standard Matrix Normal
    - [ ] Unified metric module: (Metric -> MomentMetric/DensityMetric/KernelMetric)
    - [ ] Visualization module with option for run-time smoothing.
    - [ ] Allow non-identity covariance structure for V (model and Elliptical sampler)



