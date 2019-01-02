"""
The Metric class handles running metrics throughout a simulation.

To define a Metric, it needs only a name and an update function. The update function will access any of:
    environment.action_history
    meta_vars
    environment.state
"""


class Metric(object):
    def __init__(self, update_func):
        # update_func should be a callable that takes a dict of parameters
        self.update_func = update_func
        self.vals = []

    def update(self, update_args):
        """

        :param update_args: Dict of parameters. Schema:
            action_history: List of lists of dicts corresponding to actions taken by agents in the environment
            environment_state: Current state of the environment
            meta_vars (Optional): Any meta variables global to the simulation
        :return:
        """
        self.vals.append(self.update_func(update_args))
