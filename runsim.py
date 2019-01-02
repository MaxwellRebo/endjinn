import argparse
import imp
import os
import json
import numpy as np
import marktime as mt
import endjinn
import endjinn.registry.agent_registry
import endjinn.registry.environment_registry
import endjinn.registry.action_registry
from endjinn.ml.policy import FFPolicy
from endjinn.ml.es import CMAES

parser = argparse.ArgumentParser(description='Simulation engine with lots of stuff.')
parser.add_argument('--file', default="endjinnfile.json")
parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
args = parser.parse_args()

DEFAULT_SOLVER_POPSIZE = 250

if __name__ == "__main__":
    with open(args.file) as f:
        sim_params = json.load(f)

    with open('global_registry.json') as f:
        global_registry = json.load(f)

    with open('local_registry.json') as f:
        local_registry = json.load(f)

    # Check params are correctly structured
    endjinn.check_params(sim_params)
    env_name = sim_params["environment"]["name"]

    if not env_name:
        raise Exception("Environment name must be specified in endjinn file.")

    env_file_path = None
    env_registry_entry = endjinn.registry.environment_registry.get_registry_entry(env_name, global_registry,
                                                                                  local_registry)
    file_list = os.listdir(os.getcwd() + "/registry_objects")

    if env_registry_entry["file"] in file_list:
        env_file_path = os.getcwd() + "/registry_objects/" + env_registry_entry["file"]
    else:
        file_list = os.listdir(os.getcwd() + "/local_registry_objects")

        if env_registry_entry["file"] in file_list:
            env_file_path = os.getcwd() + "/local_registry_objects/" + env_registry_entry["file"]

    if not env_file_path:
        raise Exception("Environment file %s not found." % env_registry_entry["file"])

    # env_module = importlib.import_module(env_file_path)
    env_module = imp.load_source(env_name, env_file_path)
    env = getattr(env_module, env_registry_entry["classname"])(*sim_params["environment"]["args"])

    print "\nLoaded environment %s from %s\n" % (env_name, env_file_path)

    population = sim_params.get("population")
    agent_modules = {}
    agent_registry_entries = {}
    agent_reward_vars = {}

    # Check that each agent has an entry
    for agent in sim_params["agents"]:
        if not endjinn.registry.agent_registry.check_for_agent(agent["name"], global_registry, local_registry):
            raise Exception("Agent %s not found in registry. Check the agent name." % agent["name"])
        else:
            agent_reward_vars[agent["name"]] = agent["reward_var"]
            agent_registry_entries[agent["name"]] = endjinn.registry.agent_registry.get_registry_entry(agent["name"],
                                                                                                       global_registry,
                                                                                                       local_registry)
            file_path = None
            # Look in global registry first
            file_list = os.listdir(os.getcwd() + "/registry_objects")

            if agent_registry_entries[agent["name"]]["file"] in file_list:
                file_path = os.getcwd() + "/registry_objects/" + agent_registry_entries[agent["name"]]["file"]
            else:
                # Check local registry
                file_list = os.listdir(os.getcwd() + "/local_registry_objects")

                if agent_registry_entries[agent["name"]]["file"] in file_list:
                    file_path = os.getcwd() + "/local_registry_objects/" + agent_registry_entries[agent["name"]]["file"]

            if not file_path:
                raise Exception("Agent file %s not found." % agent_registry_entries[agent["name"]]["file"])

            agent_modules[agent["name"]] = imp.load_source(agent["name"], file_path)
            print "\nLoaded agent %s from %s\n" % (agent["name"], file_path)

    agent_types = []
    agent_registry_entries_by_classname = {}
    policies = []
    solvers = []
    param_lengths = []
    action_lengths = []
    action_sets = {}
    action_modules = {}
    action_param_sets = {}
    agent_pops = {}
    all_actions = []

    for agent in sim_params["agents"]:
        if agent["pop_percentage"]:
            agent_pop = int(agent["pop_percentage"] * population)
        else:
            agent_pop = int(population / len(sim_params["agents"]))

        agent_pops[agent["name"]] = agent_pop
        ag = getattr(agent_modules[agent["name"]], agent_registry_entries[agent["name"]]["classname"])(*agent["args"])
        param_lengths.append(ag.varmap.input_length)
        action_lengths.append(len(ag.actions))
        agent_types.append(agent_registry_entries[agent["name"]]["classname"])
        action_sets[agent_registry_entries[agent["name"]]["classname"]] = ag.actions
        all_actions += ag.actions
        agent_registry_entries_by_classname[agent_registry_entries[agent["name"]]["classname"]] = \
            agent_registry_entries[agent["name"]]

        for act in agent_registry_entries[agent["name"]]["actions"]:
            if "param_attributes" in act:
                action_param_sets[act["name"]] = act["param_attributes"]
            else:
                action_param_sets[act["name"]] = []

    all_actions = set(all_actions)

    for action in all_actions:
        registry_entry = endjinn.registry.action_registry.get_registry_entry(action, global_registry, local_registry)

        file_path = None
        # Look in global registry first
        file_list = os.listdir(os.getcwd() + "/registry_objects")

        if registry_entry["file"] in file_list:
            file_path = os.getcwd() + "/registry_objects/" + registry_entry["file"]
        else:
            # Check local registry
            file_list = os.listdir(os.getcwd() + "/local_registry_objects")

            if registry_entry["file"] in file_list:
                file_path = os.getcwd() + "/local_registry_objects/" + registry_entry["file"]

        if not file_path:
            raise Exception("Action file %s not found." % registry_entry["file"])

        action_modules[action] = imp.load_source(action, file_path)
        print "\nLoaded action %s from %s\n" % (action, file_path)

    for i, nparams in enumerate(param_lengths):
        policies.append(FFPolicy(nparams, action_lengths[i]))
        solvers.append(CMAES(policies[-1].total_params,
                             popsize=DEFAULT_SOLVER_POPSIZE,
                             weight_decay=0.0,
                             sigma_init=0.5
                             ))

    # Run simulation
    policy_reward_history = []
    runs = sim_params["runs"]

    for i in range(runs):
        mt.start("run")
        solns = []
        policy_rewards = [None] * len(policies)

        for j, solver in enumerate(solvers):
            solns.append(solver.ask())

        all_fitnesses = []

        if args.verbose:
            print "Starting run %i" % i

        for solver in solvers:
            all_fitnesses.append(np.zeros(DEFAULT_SOLVER_POPSIZE))

        for k in range(DEFAULT_SOLVER_POPSIZE):
            # Re-instantiate env for a clean run for each solution
            env = getattr(env_module, env_registry_entry["classname"])(*sim_params["environment"]["args"])
            agents = []

            for agent in sim_params["agents"]:
                for k in range(agent_pops[agent["name"]]):
                    agt = getattr(agent_modules[agent["name"]], agent_registry_entries[agent["name"]]["classname"])(
                        *agent["args"])
                    agt.set_get_env(env)
                    agt.name = agent["name"]
                    agents.append(agt)

            for j, soln in enumerate(solns):
                policies[j].set_model_weights(soln[k])

            # Run complete number of ticks
            for tick in range(sim_params["ticks_per_run"]):
                for n, agent in enumerate(agents):
                    policy_index = agent_types.index(agent.__class__.__name__)
                    agent_inp = agent.get_input()
                    agent_inp = agent_inp.ravel()[None]
                    pred = policies[policy_index].single_prediction(agent_inp)
                    final_pred = None

                    if len(pred) == 1:
                        if pred[0] > 0:
                            final_pred = 1
                        elif pred[0] <= 0:
                            final_pred = 0
                    else:
                        final_pred = np.argmax(pred)

                    action = None
                    action_name = None

                    if len(pred) == 1:
                        # One action
                        if final_pred == 1:
                            action = action_modules[agent.actions[0]].handler
                            action_name = agent.actions[0]
                    else:
                        # Multiple actions
                        action_name = agent.actions[final_pred]
                        action = action_modules[action_name].handler

                    if action_name and action:
                        action_params = {}

                        for thing in action_param_sets[action_name]:
                            action_params[thing] = getattr(agent, thing)

                        update_dict = action(agent.state, env.state, action_params)
                        update_dict.apply(agent.state)
                        env.register_action(n, action_name, action_params)

                env._tick()

            # Gather rewards
            temp_fitnesses = []

            for s in range(len(solvers)):
                temp_fitnesses.append([])

            for agent in agents:
                solver_index = agent_types.index(agent.__class__.__name__)
                agent_reward = agent.state[agent_reward_vars[agent.name]]
                temp_fitnesses[solver_index].append(agent_reward)

            for f, thing in enumerate(temp_fitnesses):
                avg = np.mean(thing)
                all_fitnesses[f][k] = avg

        best_fitnesses = []

        for g, thing in enumerate(all_fitnesses):
            solvers[g].tell(thing)
            best_fitnesses.append(solvers[g].result()[1])

        duration = mt.stop("run")

        if args.verbose:
            print "Run %i complete in %f seconds" % (i, duration.seconds)
            print "Best fitnesses in run:", best_fitnesses
