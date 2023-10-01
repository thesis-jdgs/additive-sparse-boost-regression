"""Define helper classes for sampling."""
import abc

import attrs
import numpy as np


@attrs.define
class SampleGenerator(abc.ABC):
    """Generate sub-samples for each iteration."""

    size: int = attrs.field()
    fraction: float = attrs.field()
    random_generator: np.random.Generator = attrs.field()

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        ...


@attrs.define
class DeterministicGenerator(SampleGenerator):
    """Generate deterministic samples."""

    def __call__(self) -> np.ndarray:
        """Generate a deterministic sample."""
        return np.ones(self.size, dtype=np.float64)


@attrs.define
class BootstrapSampleGenerator(SampleGenerator):
    """Generate bootstrap samples."""

    def __attrs_post_init__(self):
        self._n_trials = int(self.size * self.fraction)
        self._p = 1 / self.size

    def __call__(self) -> np.ndarray:
        """Generate a bootstrap sample."""
        return self.random_generator.binomial(self._n_trials, self._p, self.size)


@attrs.define
class PoissonSampleGenerator(SampleGenerator):
    """Generate Poisson samples, which approximate the Boostrap distribution."""

    def __call__(self) -> np.ndarray:
        """Generate a Poisson sample."""
        return self.random_generator.poisson(self.fraction, self.size)


@attrs.define
class SubSampleGenerator(SampleGenerator):
    """Generate sub-samples without replacement."""

    def __call__(self) -> np.ndarray:
        """Generate a sub-sample."""
        return (self.random_generator.random(self.size) <= self.fraction).astype(
            np.float64
        )


generator_dict = {
    "subsample": SubSampleGenerator,
    "mini-batch": SubSampleGenerator,
    "bernoulli": SubSampleGenerator,
    "bootstrap": BootstrapSampleGenerator,
    "binomial": BootstrapSampleGenerator,
    "poisson": PoissonSampleGenerator,
    "none": DeterministicGenerator,
    None: DeterministicGenerator,
}
