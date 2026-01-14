import numpy as np
import matplotlib.pyplot as plt
import time
import sympy as sp
from itertools import chain
# Basic math
from sympy import exp, sin, cos, oo, pi, log
# Continuous distributions
from sympy.stats import (Normal, Uniform, Exponential, Gamma, Beta, ChiSquared, StudentT, FDistribution, Cauchy,
                         Rayleigh, LogNormal, Weibull, Pareto, Gompertz, Laplace, Maxwell, PowerFunction, Triangular,
                         ContinuousRV)
# Discrete distributions
from sympy.stats import (Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial, Hypergeometric, DiscreteUniform,
                         Zeta, Rademacher)
# Operators and functions
from sympy.stats import (density, E, H, P, variance, covariance, cdf, sample, moment, cmoment, smoment,
                         moment_generating_function, factorial_moment, skewness, coskewness, kurtosis,
                         characteristic_function)


EXCLUDED = (str, bytes, bytearray, set, frozenset)


def is_sequence_like(obj):
    from collections.abc import Sequence
    import numpy as np

    return (
        isinstance(obj, Sequence)
        and not isinstance(obj, EXCLUDED)
    ) or isinstance(obj, np.ndarray)


class RandomSample:
    def __init__(self, rvs, x='x', n='n'):
        """
        Always assumes independent random variables.
        :param rvs: if single random variable, it assumes n iid rvs, otherwise finite number
        :param x: name of x-variable
        :param n: name of n-variable
        """
        if is_sequence_like(rvs):  # If rvs is sequence-like
            self.seq = True
            self.n = sp.Integer(len(rvs))
            self.prob_functions = [density(rv) for rv in rvs]
            self.rvs = list(rvs)
            self.x = sp.symbols([rv.name for rv in self.rvs])
            self.params = list(set(chain.from_iterable([prob_func.free_symbols for prob_func in self.prob_functions])))
            self.subs_vars = {str(symbol): rv for symbol, rv in zip(self.x, self.rvs)}  # dict for substitutions
        else:  # If rvs is single rv (=> identically distributed)
            self.seq = False
            self.n = sp.Symbol(n, integer=True, positive=True)
            self.prob_functions = density(rvs)
            self.rvs = rvs
            self.params = list(self.prob_functions.free_symbols)
            self.i = sp.Symbol('i', integer=True, positive=True)
            self.x = sp.IndexedBase(x)

    def likelihood(self, expand=True):
        if self.seq:
            pdfs = [prob_function(self.rvs[finite_index]) for finite_index, prob_function in enumerate(self.prob_functions)]
            likelihood = sp.Mul(*pdfs).subs(self.subs_vars)
        else:
            pdf = self.prob_functions(self.x[self.i])
            likelihood = sp.Product(pdf, (self.i, 1, self.n))
        if expand:
            likelihood = sp.expand_log(likelihood, force=True).expand().doit()
        return likelihood

    def log_likelihood(self, expand=True):
        if self.seq:
            pdfs = [prob_function(self.x[finite_index]) for finite_index, prob_function in enumerate(self.prob_functions)]
            log_likelihood = sp.Add(*map(log, pdfs)).subs(self.subs_vars)
        else:
            pdf = self.prob_functions(self.x[self.i])
            log_likelihood = sp.Sum(log(pdf), (self.i, 1, self.n))
        if expand:
            log_likelihood = sp.expand_log(log_likelihood, force=True).expand().doit()
        return log_likelihood

    def mle(self, parameters=None):
        if parameters is None:
            parameters = self.params
        log_likelihood = self.log_likelihood(expand=True)
        if is_sequence_like(parameters):
            ll_diff_dict = {param: sp.diff(log_likelihood, param) for param in parameters}
            mle = {param: sp.solve(ll_diff, param) for param, ll_diff in ll_diff_dict.items()}
        else:
            ll_diff = sp.diff(log_likelihood, parameters)
            mle = sp.solve(ll_diff, parameters)
        return mle

    def fisher_information(self, parameters=None):
        if parameters is None:
            parameters = self.params
        if self.seq:
            log_likelihood = self.log_likelihood()
            # score vector (gradient of log-likelihood)
            score_vec = sp.Matrix([sp.diff(log_likelihood, theta).simplify() for theta in parameters])
            return E(score_vec * score_vec.T)
        else:
            pdf = self.prob_functions(self.rvs)
            log_likelihood = log(pdf)
            score_vec = sp.Matrix([sp.diff(log_likelihood, theta).simplify() for theta in parameters])
            fi_one = sp.simplify(E(score_vec * score_vec.T))
            return self.n * fi_one

theta = sp.Symbol('theta', real=True, positive=True)
n = sp.Symbol('n', integer=True, positive=True)
x = sp.Symbol('x', real=True, positive=True)

X = Gamma('X', 4, theta)

f_X = density(X)
print(density(Normal("X", 0, 1))(x))



