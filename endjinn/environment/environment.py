class Environment(object):
    def __init__(self):
        self.pre_cycle_hooks = []
        self.post_cycle_hooks = []
        self.cycles = 0
        self.state = {}
        self.state_vars = None
        self.action_history = [list()]
        self.end = False
        self.metrics = []

    def register_pre_cycle_hook(self, cb):
        if callable(cb):
            self.pre_cycle_hooks.append(cb)

    def register_post_cycle_hook(self, cb):
        if callable(cb):
            self.post_cycle_hooks.append(cb)

    def _tick(self):
        if not self.state_vars:
            raise Exception("State vars not set. Call env.set_state_vars() in subclass init before trying to run.")

        for cb in self.pre_cycle_hooks:
            cb(self)

        self.action_history.append(list())
        self.tick_update()

        for cb in self.post_cycle_hooks:
            cb(self)

        for metric in self.metrics:
            metric.update({'action_history': self.action_history, 'environment_state': self.state})

        self.cycles += 1

    def tick_update(self):
        # tick_update must be implemented in subclass
        raise NotImplementedError()

    def update_state(self, state):
        for key, val in state.iteritems():
            if key in self.state or key in self.state_vars:
                self.state[key] = state[key]

    def set_state_vars(self, state_vars):
        for key, val in state_vars.iteritems():
            assert val in ["float", "int", "string"]

        self.state_vars = state_vars

    def register_action(self, agent_id, action, action_params=None):
        self.action_history[-1].append({
            "agent_id": agent_id,
            "action": action,
            "action_params": action_params
        })

    def add_metric(self, metric):
        self.metrics.append(metric)
