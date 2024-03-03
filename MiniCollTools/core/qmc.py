"""Quasi-Monte Carlo engines and helpers."""
from __future__ import annotations

import copy
import numbers
from abc import ABC, abstractmethod
import math
from typing import (
    Optional,
    overload,
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import GeneratorType, IntNumber, SeedType

import scipy.stats as stats


@overload
def check_random_state(seed: Optional[IntNumber] = ...) -> np.random.Generator:
    ...


@overload
def check_random_state(seed: GeneratorType) -> GeneratorType:
    ...


# Based on scipy._lib._util.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.
    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.
    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        if not hasattr(np.random, "Generator"):
            # This can be removed once numpy 1.16 is dropped
            msg = (
                "NumPy 1.16 doesn't have Generator, use either "
                "NumPy >= 1.17 or `seed=np.random.RandomState(seed)`"
            )
            raise ValueError(msg)
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # The two checks can be merged once numpy 1.16 is dropped
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.Generator" " instance" % seed
        )


class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.
    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.
    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator (used for the
    ``seed``).
    **Subclassing**
    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.
    * ``__init__(d, seed=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``seed`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``random(n)``: draw ``n`` from the engine and increase the counter
      ``num_generated`` by ``n``.
    Optionally, two other methods can be overwritten by subclasses:
    * ``reset``: Reset the engine to it's original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.
    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:
    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def random(self, n=1):
    ...         self.num_generated += n
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self
    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.
    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])
    We can also reset the state of the generator and resample again.
    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])
    """

    @abstractmethod
    def __init__(self, d: IntNumber, *, seed: SeedType = None) -> None:
        if not np.issubdtype(type(d), np.integer):
            raise ValueError("d must be an integer value")

        self.d = d
        self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(seed)
        self.num_generated = 0

    @abstractmethod
    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            QMC sample.
        """
        # self.num_generated += n

    def reset(self) -> QMCEngine:
        """Reset the engine to base state.
        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.
        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n: IntNumber) -> QMCEngine:
        """Fast-forward the sequence by `n` positions.
        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.
        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.
        """
        self.random(n=n)
        return self


class LatinHypercube(QMCEngine):
    """Latin hypercube sampling (LHS).
    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.
    LHS is extremely effective on integrands that are nearly additive [2]_.
    LHS on :math:`n` points never has more variance than plain MC on
    :math:`n-1` points [3]_. There is a central limit theorem for LHS [4]_,
    but not necessarily for optimized LHS.
    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    centered : bool, optional
        Center the point within the multi-dimensional grid. Default is False.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code",
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    Examples
    --------
    Generate samples from a Latin hypercube generator.
    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.1545328 , 0.53664833],  # random
           [0.84052691, 0.06474907],
           [0.52177809, 0.93343721],
           [0.68033825, 0.36265316],
           [0.26544879, 0.61163943]])
    Compute the quality of the sample using the discrepancy criterion.
    >>> qmc.discrepancy(sample)
    0.019558034794794565  # random
    Finally, samples can be scaled to bounds.
    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample, l_bounds, u_bounds)
    array([[1.54532796, 3.609945  ],  # random
           [8.40526909, 2.1942472 ],
           [5.2177809 , 4.80031164],
           [6.80338249, 3.08795949],
           [2.65448791, 3.83491828]])
    """

    def __init__(
        self, d: IntNumber, *, centered: bool = False, seed: SeedType = None
    ) -> None:
        super().__init__(d=d, seed=seed)
        self.centered = centered

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            LHS sample.
        """
        if self.centered:
            samples = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))  # type: ignore[assignment]

        perms = np.tile(np.arange(1, n + 1), (self.d, 1))
        for i in range(self.d):  # type: ignore[arg-type]
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        samples = (perms - samples) / n
        self.num_generated += n
        return samples  # type: ignore[return-value]


class MultivariateNormalQMC(QMCEngine):
    r"""QMC sampling from a multivariate Normal :math:`N(\mu, \Sigma)`.
    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> engine = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = engine.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()
    """

    def __init__(
        self,
        mean: npt.ArrayLike,
        cov: Optional[npt.ArrayLike] = None,
        *,
        cov_root: Optional[npt.ArrayLike] = None,
        inv_transform: bool = True,
        engine: Optional[QMCEngine] = None,
        seed: SeedType = None
    ) -> None:
        mean = np.array(mean, copy=False, ndmin=1)
        d = mean.shape[0]
        if cov is not None:
            # covariance matrix provided
            cov = np.array(cov, copy=False, ndmin=2)
            # check for square/symmetric cov matrix and mean vector has the
            # same d
            if not mean.shape[0] == cov.shape[0]:
                raise ValueError("Dimension mismatch between mean and " "covariance.")
            if not np.allclose(cov, cov.transpose()):
                raise ValueError("Covariance matrix is not symmetric.")
            # compute Cholesky decomp; if it fails, do the eigen decomposition
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1.0e-8):
                    raise ValueError("Covariance matrix not PSD.")
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
        elif cov_root is not None:
            # root decomposition provided
            cov_root = np.atleast_2d(cov_root)
            if not mean.shape[0] == cov_root.shape[0]:
                raise ValueError("Dimension mismatch between mean and " "covariance.")
        else:
            # corresponds to identity covariance matrix
            cov_root = None

        super().__init__(d=d, seed=seed)
        self._inv_transform = inv_transform

        if not inv_transform:
            # to apply Box-Muller, we need an even number of dimensions
            engine_dim = 2 * math.ceil(d / 2)
        else:
            engine_dim = d
        if engine is None:
            self.engine = LatinHypercube(d=engine_dim, seed=seed)  # type: QMCEngine
        elif isinstance(engine, QMCEngine) and engine.d != 1:
            if engine.d != d:
                raise ValueError(
                    "Dimension of `engine` must be consistent"
                    " with dimensions of mean and covariance."
                )
            self.engine = engine
        else:
            raise ValueError(
                "`engine` must be an instance of "
                "`scipy.stats.qmc.QMCEngine` or `None`."
            )

        self._mean = mean
        self._corr_matrix = cov_root

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multivariate Normal.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            Sample.
        """
        base_samples = self._standard_normal_samples(n)
        self.num_generated += n
        return self._correlate(base_samples)

    def reset(self) -> MultivariateNormalQMC:
        """Reset the engine to base state.
        Returns
        -------
        engine : MultivariateNormalQMC
            Engine reset to its base state.
        """
        super().reset()
        self.engine.reset()
        return self

    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            return base_samples @ self._corr_matrix + self._mean
        else:
            # avoid multiplying with identity here
            return base_samples + self._mean

    def _standard_normal_samples(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            Sample.
        """
        # get base samples
        samples = self.engine.random(n)
        if self._inv_transform:
            # apply inverse transform
            # (values to close to 0/1 result in inf values)
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]
        else:
            # apply Box-Muller transform (note: indexes starting from 1)
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            return transf_samples[:, : self.d]  # type: ignore[misc]
