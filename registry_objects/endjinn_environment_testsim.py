from endjinn.environment import Environment
from endjinn.state_block import Series
from endjinn.metric import Metric
import numpy as np

"""
This example environment shows use of:
    - tick_update() implementation
    - Series state block for time series
    - Defining a Metric update function and adding a metric to the environment
    
The environment simulates a single asset price using Gaussian noise as percentage change per tick. That is, every tick
the price will change by +/- 0-1%, give or take a little. This is an abstraction over potential market activity that
could be generating those price changes. The environment assumes that agent actions do not have the capability to
move the price of the underlying asset.
"""


def num_actions_uf(update_params):
    return len(update_params["action_history"][-1])


class TestSim(Environment):
    def __init__(self, std=0.02):
        super(TestSim, self).__init__()
        self.std = std
        self.state = {
            "asset_price": 1.0
        }
        self.set_state_vars({
            "asset_price": "float"
        })
        self.asset_price_history = Series([1.0])
        self.asset_perc_diffs = Series()

        num_actions = Metric(num_actions_uf)
        self.add_metric(num_actions)

        # Burn in price to build up history
        for i in range(10):
            self.iterate_price()

    def tick_update(self):
        # sample Gaussian noise, add to 1.0 to function as percentage change
        self.iterate_price()

    def iterate_price(self):
        perc = 1. + np.random.normal(0, self.std, 1)[0]
        self.state['asset_price'] = self.state['asset_price'] * perc
        self.asset_price_history.add_value(self.state['asset_price'])
        self.asset_perc_diffs.add_value(perc)
