import numpy as np
import pickle
from syne_tune.config_space import Categorical, Domain
from whittle.sampling.random_sampler import RandomSampler as WhittleRandomSampler
from whittle.metrics.parameters import (
    compute_parameters,
    compute_parameters_sub_network_gpt,
)

class RandomSampler:
    def __init__(self, search_space, seed: int | None = None):
        self.search_space = search_space
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)

    def sample(self):
        return self.search_space.cast(self.sampler.sample())

    def get_smallest_sub_network(self):
        return self.search_space.cast(self.sampler.get_smallest_sub_network())

    def get_medium_sub_network(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = hp.categories[len(hp.categories) // 2]
            else:
                u = hp.upper
                l = hp.lower
            config[hp_name] = int(0.5 * (u - l) + l)
        return self.search_space.cast(config)

    def get_largest_sub_network(self):
        return self.search_space.cast(self.sampler.get_largest_sub_network())

