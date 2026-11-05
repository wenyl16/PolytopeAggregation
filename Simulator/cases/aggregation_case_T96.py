"""96-interval aggregation case."""

from .aggregation import Aggregator as BaseAggregator


class Aggregator(BaseAggregator):
    """Resource generator with the T96 case defaults."""

    def __init__(self, seed=None, T=96, data=None, discrete_rate=0.0):
        super().__init__(seed=seed, T=T, data=data, discrete_rate=discrete_rate)
