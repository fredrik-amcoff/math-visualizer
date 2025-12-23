import numpy as np
import matplotlib.pyplot as plt
import time
import sympy as sp
# Continuous distributions
from sympy.stats import (Normal, Uniform, Exponential, Gamma, Beta, ChiSquared, StudentT, FDistribution, Cauchy,
                         Rayleigh, LogNormal, Weibull, Pareto, Gompertz, Laplace, Maxwell, PowerFunction, Triangular)
# Discrete distributions
from sympy.stats import (Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial, Hypergeometric, DiscreteUniform,
                         Zeta, Rademacher)
# Operators and functions
from sympy.stats import (density, E, H, P, variance, covariance, cdf, sample, moment, cmoment, smoment,
                         moment_generating_function, factorial_moment, skewness, coskewness, kurtosis,
                         characteristic_function)

theta = sp.Symbol('theta', real=True, positive=True)
n = sp.Symbol('n', integer=True, positive=True)
x = sp.Symbol('x', real=True, positive=True)

X = Gamma('X', 4, theta)

f_X = density(X)
print(density(Normal("X", 0, 1))(x))



