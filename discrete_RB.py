"""
This file defines the class for discrete_RB simulation environment,
the helper class for solving single-armed LPs and getting priorities,
classes for RB policies,
along with a few helper functions
"""

import numpy as np
import cvxpy as cp
import scipy


class RB(object):
    """
    the simulation environment of restless bandits
    :param sspa_siz: the size of the state space S
    :param trans_tensor: np array of shape (S,A,S), representing the transition kernel {P(s,a,s')}
    :param reward_tensor: np array of shape (S,a), representing the reward function {r(s,a)}
    :param N: number of arms
    :param init_states: the initial states of the arms. Initialize all arms to state 0 if not provided
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor, N, init_states=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = 2
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]
        self.N = N
        # map of state action to next states
        self.trans_dict = {sa_pair: trans_tensor[sa_pair[0], sa_pair[1], :] for sa_pair in self.sa_pairs}
        self.reward_dict = {sa_pair: reward_tensor[sa_pair[0], sa_pair[1]] for sa_pair in self.sa_pairs}
        # initialize the state of the arms at 0
        if init_states is not None:
            self.states = init_states.copy()
        else:
            self.states = np.zeros((self.N,))

    def get_states(self):
        return self.states.copy()

    def step(self, actions):
        """
        :param actions: a 1-d array with length N. Each entry is 0 or 1, denoting the action of each arm
        :return: intantaneous reward of the this time step
        """
        sa2indices = {sa_pair:None for sa_pair in self.sa_pairs} # each key is a list of indices in the range of self.N
        # find out the arms whose (state, action) = sa_pair
        for sa_pair in self.sa_pairs:
            sa2indices[sa_pair] = np.where(np.all([self.states == sa_pair[0], actions == sa_pair[1]], axis=0))[0]
        instant_reward = 0
        for sa_pair in self.sa_pairs:
            next_states = np.random.choice(self.sspa, len(sa2indices[sa_pair]), p=self.trans_dict[sa_pair])
            self.states[sa2indices[sa_pair]] = next_states
            instant_reward += self.reward_dict[sa_pair] * len(sa2indices[sa_pair])
        instant_reward = instant_reward / self.N  # we normalize it by the number of arms
        return instant_reward

    def get_s_counts(self):
        s_counts = np.zeros(self.sspa_size)
        # find out the arms whose (state, action) = sa_pair
        for state in self.sspa:
            s_counts[state] = len(np.where([self.states == state])[0])
        return s_counts

    def get_s_fracs(self):
        s_counts = self.get_s_counts()
        s_fracs = np.zeros(self.sspa_size)
        for state in self.sspa:
            s_fracs[state] = s_counts[state] / self.N
        return s_fracs


class MeanFieldRB(object):
    """
    RB with infinite many arms that transition according to the mean-field dynamics
    :param sspa_siz: the size of the state space S
    :param trans_tensor: np array of shape (S,A,S), representing the transition kernel {P(s,a,s')}
    :param reward_tensor: np array of shape (S,a), representing the reward function {r(s,a)}
    :param init_state_fracs: the initial fraction of arms in each state. Initialize all arms to state 0 if not provided
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor, init_state_fracs=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = 2
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.EPS = 1e-7  # numbers smaller than this are regard as zero

        # states are represented by the distribution of the arms
        if init_state_fracs is None:
            self.state_fracs = np.zeros((self.sspa_size,))
            self.state_fracs[0] = 1
        else:
            self.state_fracs = init_state_fracs.copy()

    def step(self, sa_pair_fracs):
        """
        :param sa_pair_fracs:
        :return: intantaneous reward of the this time step
        """
        assert np.all(np.isclose(np.sum(sa_pair_fracs, axis=1), self.state_fracs, atol=1e-4)), \
            "the input sa_pair_fracs is not consistent with current state, {}!={}".format(np.sum(sa_pair_fracs, axis=1), self.state_fracs)
        new_state_fracs = np.zeros((self.sspa_size))
        instant_reward = 0
        for new_state in self.sspa:
            new_state_fracs[new_state] = np.sum(sa_pair_fracs * self.trans_tensor[:,:,new_state])
        #print(sa_pair_fracs * self.reward_tensor)
        instant_reward += np.sum(sa_pair_fracs * self.reward_tensor)
        assert np.isclose(np.sum(new_state_fracs), 1.0, atol=1e-4), "new state fractions do not sum to one, the number we get is {}".format(np.sum(new_state_fracs))
        self.state_fracs = new_state_fracs
        return instant_reward

    def get_state_fracs(self):
        return self.state_fracs.copy()


class SingleArmAnalyzer(object):
    """
    solving the single-armed problem
    :param sspa_siz: the size of the state space S
    :param trans_tensor: np array of shape (S,A,S), representing the transition kernel {P(s,a,s')}
    :param reward_tensor: np array of shape (S,a), representing the reward function {r(s,a)}
    :param act_frac: the fraction of arms to activate in each time slot
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor, act_frac):
        # problem parameters
        self.sspa_size = sspa_size   # state-space size
        self.sspa = np.array(list(range(self.sspa_size)))  # state space
        self.aspa_size = 2   # action-space size
        self.aspa = np.array(list(range(self.aspa_size)))  # action space
        self.sa_pairs = []   # all possible combinations of (state,action), defined for the convenience of iteration
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.act_frac = act_frac
        # some constants
        self.EPS = 1e-7  # any numbers smaller than this are regard as zero
        min_reward = np.min(self.reward_tensor)
        max_reward = np.max(self.reward_tensor)
        self.DUALSTEP = 0.05 # the discretization step size of dual variable when solving for Whittle's index
        self.MINSUBSIDY = - (max_reward - min_reward) - self.DUALSTEP # a lower bound on the set of possible subsidies
        self.MAXSUBSIDY = (max_reward - min_reward) + self.DUALSTEP  # an upper bound on the set of possible subsidies

        # variables
        self.y = cp.Variable((self.sspa_size, 2))
        self.dualvar = cp.Parameter(name="dualvar")  # the subsidy parameter for solving Whittle's index policy

        # store some data of the solution, only needed for solving the LP-Priority policy
        # the values might change, so they are not safe to use unless immediately after solving the LP.
        self.opt_value = None
        self.avg_reward = None
        self.opt_subsidy = None
        self.value_func_relaxed = np.zeros((self.sspa_size,))
        self.q_func_relaxed = np.zeros((self.sspa_size, 2))

    def get_stationary_constraints(self):
        stationary_constrs = []
        for cur_s in self.sspa:
            m_s = cp.sum(cp.multiply(self.y, self.trans_tensor[:,:,cur_s]))
            stationary_constrs.append(m_s == cp.sum(self.y[cur_s, :]))
        return stationary_constrs

    def get_budget_constraints(self):
        budget_constrs = []
        budget_constrs.append(cp.sum(self.y[:,1]) == self.act_frac)
        return budget_constrs

    def get_basic_constraints(self):
        # the constraints that make sure we solve a probability distribution
        basic_constrs = []
        basic_constrs.append(self.y >= 1e-8)
        basic_constrs.append(cp.sum(self.y) == 1)
        return basic_constrs

    def get_objective(self):
        objective = cp.Maximize(cp.sum(cp.multiply(self.y, self.reward_tensor)))
        return objective

    def get_relaxed_objective(self):
        # the objective for the relaxed problem
        subsidy_reward_1 = self.reward_tensor[:,1]
        subsidy_reward_0 = self.reward_tensor[:,0] + self.dualvar
        relaxed_objective = cp.Maximize(cp.sum(cp.multiply(self.y[:,1], subsidy_reward_1))
                                        + cp.sum(cp.multiply(self.y[:,0], subsidy_reward_0)))
        return relaxed_objective

    def solve_lp(self):
        objective = self.get_objective()
        constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        self.opt_value = problem.solve()
        print("--------LP relaxation solved, solution as below-------")
        print("Optimal value ", self.opt_value)
        print("Optimal var")
        print(self.y.value)
        print("---------------------------")
        return (self.opt_value, self.y.value)

    def print_Phi(self):
        # self.solve_lp()
        y = self.y.value

        state_probs = np.sum(y, axis=1)
        print("mu*=", state_probs)
        policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if state_probs[state] > self.EPS:
                policy[state, :] = y[state, :] / state_probs[state]
            else:
                policy[state, 0] = 0.5
                policy[state, 1] = 0.5
        print("policy=", policy)

        ind_neu = np.where(np.all([y[:,0] > self.EPS, y[:,1] > self.EPS],axis=0))[0]
        print("ind_neu=", ind_neu)

        Ppibs = np.zeros((self.sspa_size, self.sspa_size,))
        for a in self.aspa:
            Ppibs += self.trans_tensor[:,a,:]*np.expand_dims(policy[:,a], axis=1)
        Phi = Ppibs -  np.outer(np.ones((self.sspa_size,)), state_probs) \
                - np.outer(policy[:,1] - self.act_frac * np.ones((self.sspa_size,)), self.trans_tensor[ind_neu,1,:] - self.trans_tensor[ind_neu,0,:])
        moduli = [np.absolute(lam) for lam in np.linalg.eigvals(Phi)]
        spec_rad = max(moduli)

        print(policy[:,1] - self.act_frac * np.ones((self.sspa_size,)))

        print("P1=", self.trans_tensor[:,1,:])
        print("P0=", self.trans_tensor[:,0,:])
        print("Ppibs=", Ppibs)
        print("Phi=", Phi)
        print("moduli=", moduli)

    def compute_W(self, abstol):
        # self.solve_lp()
        y = self.y.value

        state_probs = np.sum(y, axis=1)
        # print("mu*=", state_probs)
        policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if state_probs[state] > self.EPS:
                policy[state, :] = y[state, :] / state_probs[state]
            else:
                policy[state, 0] = 0.5
                policy[state, 1] = 0.5
        # print("policy=", policy)

        Ppibs = np.zeros((self.sspa_size, self.sspa_size,))
        for a in self.aspa:
            Ppibs += self.trans_tensor[:,a,:]*np.expand_dims(policy[:,a], axis=1)

        Ppibs_centered = Ppibs - np.outer(np.ones((self.sspa_size,)), state_probs)
        W = np.zeros((self.sspa_size, self.sspa_size))
        P_power = np.eye(self.sspa_size)
        # calculate W, test tolerance level
        iters = 0
        while True:
            W += np.matmul(P_power, P_power.T)
            P_power = np.matmul(Ppibs_centered, P_power)
            P_power_norm = np.linalg.norm(P_power)
            W_norm = np.linalg.norm(W, ord=2)
            # print("P_power_norm=", P_power_norm, "W_norm=", W_norm)
            spn_error = W_norm * P_power_norm**2 / (1-P_power_norm**2)
            iters += 1
            if (P_power_norm < 1) and (spn_error < abstol):
                break
        print("W computed after expanding {} terms".format(iters))
        return W, spn_error

    def solve_LP_Priority(self, fixed_dual=None):
        if fixed_dual is None:
            objective = self.get_objective()
            constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        else:
            self.dualvar = fixed_dual
            objective = self.get_relaxed_objective()
            constrs = self.get_stationary_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        self.opt_value = problem.solve(verbose=False)

        # get value function from the dual variables. Later we should rewrite the dual problem explicitly
        # average reward is the dual variable of "sum to 1" constraint
        self.avg_reward = constrs[-1].dual_value     # the sign is positive, DO NOT CHANGE IT
        for i in range(self.sspa_size):
            # value function is the dual of stationary constraint
            self.value_func_relaxed[i] = - constrs[i].dual_value   # the sign is negative, DO NOT CHANGE IT
        if fixed_dual is None:
            # optimal subsidy for passive actions is the dual of budget constraint
            self.opt_subsidy = constrs[self.sspa_size].dual_value
        else:
            self.opt_subsidy = fixed_dual

        print("lambda* = ", self.opt_subsidy)
        print("avg_reward = ", self.avg_reward)
        print("value_func = ", self.value_func_relaxed)

        for i in range(self.sspa_size):
            for j in range(2):
                self.q_func_relaxed[i, j] = self.reward_tensor[i, j] + self.opt_subsidy * (j==0) - self.avg_reward + np.sum(self.trans_tensor[i, j, :] * self.value_func_relaxed)
        print("q func = ", self.q_func_relaxed)
        print("action gap =  ", self.q_func_relaxed[:,1] - self.q_func_relaxed[:,0])

        priority_list = np.flip(np.argsort(self.q_func_relaxed[:,1] - self.q_func_relaxed[:,0]))
        return list(priority_list)

    def solve_whittles_policy(self):
        # solve a set of relaxed problem with different subsidy values, and find out the Whittle's index
        relaxed_objective = self.get_relaxed_objective()
        constrs = self.get_stationary_constraints() + self.get_basic_constraints()
        problem = cp.Problem(relaxed_objective, constrs)
        subsidy_values = np.arange(self.MINSUBSIDY, self.MAXSUBSIDY, self.DUALSTEP)
        passive_table = np.zeros((self.sspa_size, len(subsidy_values))) # each row is a state, each column is a dual value
        for i, subsidy in enumerate(subsidy_values):
            self.dualvar.value = subsidy
            problem.solve()
            for state in self.sspa:
                if (self.y.value[state, 0] > self.EPS) and (self.y.value[state, 1] < self.EPS):
                    passive_table[state, i] = 1
        wi2state = {}
        for state in self.sspa:
            approx_wi = np.where(passive_table[state, :])[0][0]  # find the smallest subsidy such that state becomes passive
            indexable = np.all(passive_table[state, approx_wi:] == 1)  # check indexability
            if approx_wi not in wi2state:
                wi2state[approx_wi] = [state]
            else:
                print("Warning: two states have the same Whittle's index. Breaking ties favoring smaller states")
                wi2state[approx_wi].append(state)
        # sorting from states with large index to small index
        wi2state_keys_sorted = sorted(wi2state.keys(), reverse=True)
        wi2state_sorted = {key: wi2state[key] for key in wi2state_keys_sorted}
        priority_list = []
        for approx_wi in wi2state_sorted:
            priority_list += wi2state_sorted[approx_wi]
        return priority_list, indexable


class PriorityPolicy(object):
    def __init__(self, sspa_size, priority_list, N, act_frac):
        """
        The policy that uses a certain priority to allocate budget
        :param sspa_size: size of the state space S
        :param priority_list: a list of states represented by numbers, from high priority to low priority
        :param N: number of arms
        :param act_frac: the fraction of arms to activate in each time slot
        """
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.priority_list = priority_list
        self.act_frac = act_frac
        self.N = N

    def get_actions(self, cur_states):
        """
        :param cur_states: the current states of the arms
        :return: the actions taken by the arms under the policy
        """
        # return actions from states
        s2indices = {state:None for state in self.sspa} # each key is a list of indices in the range of self.N
        # find out the arms whose (state, action) = sa_pair
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]

        actions = np.zeros((self.N,))
        rem_budget = int(self.N * self.act_frac)
        rem_budget += np.random.binomial(1, self.N * self.act_frac - rem_budget)  # randomized rounding
        # go from high priority to low priority
        for state in self.priority_list:
            num_arms_this_state = len(s2indices[state])
            if rem_budget >= num_arms_this_state:
                actions[s2indices[state]] = 1
                rem_budget -= num_arms_this_state
            else:
                # break ties uniformly, sample without replacement
                chosen_indices = np.random.choice(s2indices[state], size=rem_budget, replace=False)
                actions[chosen_indices] = 1
                rem_budget = 0
                break
        assert rem_budget == 0, "something is wrong, whittles index should use up all the budget"
        return actions

    def get_sa_pair_fracs(self, cur_state_fracs):
        sa_pair_fracs = np.zeros((self.sspa_size, 2))
        rem_budget_normalize = self.act_frac
        for state in self.priority_list:
            frac_arms_this_state = cur_state_fracs[state]
            if rem_budget_normalize >= frac_arms_this_state:
                sa_pair_fracs[state, 1] = frac_arms_this_state
                sa_pair_fracs[state, 0] = 0.0
                rem_budget_normalize -= frac_arms_this_state
            else:
                sa_pair_fracs[state, 1] = rem_budget_normalize
                sa_pair_fracs[state, 0] = frac_arms_this_state - rem_budget_normalize
                rem_budget_normalize = 0
        assert rem_budget_normalize == 0.0, "something is wrong, whittles index should use up all the budget"
        return sa_pair_fracs


class RandomTBPolicy(object):
    """
    The policy that makes decisions only based on the current real states, and break ties randomly
    :param sspa_size: size of the state space S
    :param y: a solution of the single-armed problem
    :param N: number of arms
    :param act_frac: the fraction of arms to activate in each time slot
    """
    def __init__(self, sspa_size, y, N, act_frac):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.EPS = 1e-7

        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_actions(self, cur_states):
        """
        :param cur_states: the current states of the arms
        :return: the actions taken by the arms under the policy
        """
        s2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]
        actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])

        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding
        num_requests = np.sum(actions)
        if num_requests > budget:
            indices_request = np.where(actions==1)[0]
            request_ignored = np.random.choice(indices_request, int(num_requests - budget), replace=False)
            actions[request_ignored] = 0
        else:
            indices_no_request = np.where(actions==0)[0]
            no_request_pulled = np.random.choice(indices_no_request, int(budget - num_requests), replace=False)
            actions[no_request_pulled] = 1

        return actions


class FTVAPolicy(object):
    """
    FTVA policy
    :param sspa_size: size of the state space S S
    :param trans_tensor: np array of shape (S,A,S), representing the transition kernel {P(s,a,s')}
    :param reward_tensor: np array of shape (S,a), representing the reward function {r(s,a)}
    :param y: a solution of the single-armed problem
    :param N: number of arms
    :param act_frac: the fraction of arms to activate in each time slot
    :param init_virtual: initial virtual states of the arms; initialized uniformly at random if not provided
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor, y, N, act_frac, init_virtual):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.EPS = 1e-7

        if init_virtual is None:
            self.virtual_states = np.random.choice(self.sspa, self.N)
        else:
            self.virtual_states = init_virtual.copy()

        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_actions(self, cur_states, tb_rule=None, tb_param=None):
        """
        :param cur_states: current state, actually not needed
        :param tb_rule: a string, "goodness" "naive" "priority" or "goodness-priority". By default it is "goodness"
        :param tb_param: parameter of the tie-breaking policy. Only needed if tb_rule = "priority" or "goodness-priority".
        :return: actions, virtual_actions
        """
        # generate budget using randomized rounding
        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding
        # generate virtual actions according to virtual states
        vs2indices = {state:None for state in self.sspa}
        # find the indices of arms that are in each virtual state
        for state in self.sspa:
            vs2indices[state] = np.where(self.virtual_states == state)[0]
        # generate virtual actions using the policy
        virtual_actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            virtual_actions[vs2indices[state]] = np.random.choice(self.aspa, size=len(vs2indices[state]), p=self.policy[state])

        # Below we modify virtual actions into real actions, so that they satisfy the budget constraint
        # several different tie breaking rules
        if (tb_rule is None) or (tb_rule == "goodness"):
            # method 1 (the default option): we prioritize maintaining "good arms"
            # then essentially four priority levels:
            # request+good, request+bad, no_req +bad, no_req +good
            actions = np.zeros((self.N,))
            good_arm_mask = cur_states == self.virtual_states
            priority_levels = [virtual_actions * good_arm_mask, virtual_actions * (1-good_arm_mask),
                                (1-virtual_actions)*(1-good_arm_mask), (1-virtual_actions)*good_arm_mask]
            rem_budget = budget
            for i in range(len(priority_levels)):
                level_i_indices = np.where(priority_levels[i])[0] # find the indices of the arms whose priority is in level i.
                if rem_budget >= len(level_i_indices):
                    actions[level_i_indices] = 1
                    rem_budget -= len(level_i_indices)
                else:
                    activate_indices = np.random.choice(level_i_indices, rem_budget, replace=False)
                    actions[activate_indices] = 1
                    rem_budget = 0
                    break
        elif tb_rule == "naive":
            ### (IMPORTANT) Here we can have different ways of select arms to respond to
            # method 2: we randomly choose arms and flip their actions
            actions = virtual_actions.copy()
            num_requests = np.sum(actions)
            if num_requests > budget:
                indices_request = np.where(virtual_actions==1)[0]
                request_ignored = np.random.choice(indices_request, int(num_requests - budget), replace=False)
                actions[request_ignored] = 0
            else:
                indices_no_request = np.where(actions==0)[0]
                no_request_pulled = np.random.choice(indices_no_request, int(budget - num_requests), replace=False)
                actions[no_request_pulled] = 1
        elif tb_rule == "priority":
            # tie-breaking based on priority of the virtual states.
            # specifically, first break into two large classes based on virtual actions
            # then within each large class, break into smaller classes using the priority
            # this should be equivalent to tie-breaking using pure priority as long as there is only one neutral state
            assert type(tb_param) == list, "tb_param should be priority list, sorted from high priority states to low priority states"
            actions = np.zeros((self.N,))
            # divide into two rough classe using virtual actions. Arms with virtual action = 1 has higher priority than those with virtual action = 0
            priority_rough_classes = [virtual_actions, 1-virtual_actions]
            priority_levels = []
            for rough_class in priority_rough_classes:
                for i in range(len(tb_param)):
                    priority_levels.append(np.where(rough_class * (self.virtual_states == tb_param[i]))[0])
            # assign budget along the priority levels
            rem_budget = budget
            for i in range(len(priority_levels)):
                # selecting arms whose virtual states rank i in the priority list
                level_i_indices = priority_levels[i]
                if rem_budget >= len(level_i_indices):
                    actions[level_i_indices] = 1
                    rem_budget -= len(level_i_indices)
                else:
                    activate_indices = np.random.choice(level_i_indices, rem_budget, replace=False)
                    actions[activate_indices] = 1
                    rem_budget = 0
                    break
        elif tb_rule == "goodness-priority":
            # tie-breaking based on the goodness of arms, and the priority of virtual states.
            # good + fluid_active > bad + fluid_active > bad + fluid_passive > good + fluid_passive
            assert type(tb_param) == list, "tb_param should be priority list, sorted from high priorities to low priorities"
            actions = np.zeros((self.N,))
            good_arm_mask = cur_states == self.virtual_states
            priority_rough_classes = [virtual_actions * good_arm_mask, virtual_actions * (1-good_arm_mask),
                                (1-virtual_actions)*(1-good_arm_mask), (1-virtual_actions)*good_arm_mask]
            priority_levels = []
            for rough_class in priority_rough_classes:
                for i in range(len(tb_param)):
                    priority_levels.append(np.where(rough_class * (self.virtual_states == tb_param[i]))[0])
            rem_budget = budget
            for i in range(len(priority_levels)):
                level_i_indices = priority_levels[i]
                if rem_budget >= len(level_i_indices):
                    actions[level_i_indices] = 1
                    rem_budget -= len(level_i_indices)
                else:
                    activate_indices = np.random.choice(level_i_indices, rem_budget, replace=False)
                    actions[activate_indices] = 1
                    rem_budget = 0
                    break
        else:
            raise NotImplementedError

        return actions, virtual_actions

    def virtual_step(self, prev_states, cur_states, actions, virtual_actions):
        # simulation with coupling
        agree_mask = np.all([prev_states == self.virtual_states, actions == virtual_actions], axis=0)
        agree_indices = np.where(agree_mask)[0]
        # for those arms whose virtual state-action pairs agree with real ones (good arms), couple the next states
        self.virtual_states[agree_indices] = cur_states[agree_indices]

        sa2indices = {sa_pair:[] for sa_pair in self.sa_pairs}
        for sa_pair in self.sa_pairs:
            # find out the indices of the bad arms whose (virtual state, action) = sa_pair
            sa2indices[sa_pair] = np.where(np.all([self.virtual_states == sa_pair[0],
                                                   virtual_actions == sa_pair[1],
                                                   1 - agree_mask], axis=0))[0]
        for sa_pair in self.sa_pairs:
            cur_indices = sa2indices[sa_pair[0], sa_pair[1]]
            self.virtual_states[cur_indices] = np.random.choice(self.sspa, size=len(cur_indices),
                                                                p=self.trans_tensor[sa_pair[0], sa_pair[1]])

    def get_virtual_states(self):
        return self.virtual_states.copy()


class IDPolicy(object):
    def __init__(self, sspa_size, y, N, act_frac):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.EPS = 1e-7

        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_action(self, cur_states):
        """
        :param cur_states: the current states of the arms
        :return: the actions taken by the arms under the policy
        """
        s2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]
        actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])

        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding when alpha N

        # todo: change the logic below, use ID rather than random TB
        num_requests = np.sum(actions)
        if num_requests > budget:
            indices_request = np.where(actions==1)[0]
            request_ignored = np.random.choice(indices_request, int(num_requests - budget), replace=False)
            actions[request_ignored] = 0
        else:
            indices_no_request = np.where(actions==0)[0]
            no_request_pulled = np.random.choice(indices_no_request, int(budget - num_requests), replace=False)
            actions[no_request_pulled] = 1

        return actions


class SetExpansionPolicy(object):
    def __init__(self, sspa_size, y, N, act_frac):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.EPS = 1e-7
        # # the focus set, represented as an array of IDs of the arms
        # self.focus_set = np.array([])

        # variables and parameters
        self.z = cp.Variable(self.sspa_size)
        self.m = cp.Variable()
        self.d = cp.Variable(self.sspa_size)
        # self.s_count_scaled = np.zeros((self.sspa_size,))
        # self.s_count_scaled_fs = np.zeros((self.sspa_size,))
        self.s_count_scaled = cp.Parameter(self.sspa_size)
        self.s_count_scaled_fs = cp.Parameter(self.sspa_size)
        self.beta = cp.Parameter()
        self.beta = min(act_frac, 1-act_frac)
        self.state_probs = cp.Parameter(self.sspa_size)


        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_new_focus_set(self, states, last_focus_set):
        """
        states: length-N vector of states
        focus_set: array of IDs for the arms in the focus set
        """
        states_fs = states[last_focus_set]
        s2indices = {s: None for s in self.sspa}
        s2indices_fs = {s: None for s in self.sspa} # state to indices map in the focus set
        s_count_scaled = np.zeros((self.sspa_size,))
        s_count_scaled_fs = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(states == s)[0]
            s2indices_fs[s] = np.where(states_fs == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
            s_count_scaled_fs[s] = len(s2indices_fs[s]) / self.N
        print("Xt([N]) = {}, Xt(D(t-1)) = {}".format(s_count_scaled, s_count_scaled_fs))
        self.s_count_scaled = s_count_scaled
        self.s_count_scaled_fs = s_count_scaled_fs

        cur_m = len(last_focus_set)/self.N
        cur_delta = self.beta*(1-cur_m) - np.linalg.norm(s_count_scaled_fs - cur_m * self.state_probs, ord=1)

        constrs = []
        if cur_delta > 0:
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= self.s_count_scaled_fs)
        else:
            constrs.append(self.z <= self.s_count_scaled_fs)
            constrs.append(self.z >= 0)
        constrs.append(self.m == cp.sum(self.z))
        constrs.append(self.z - self.m*self.state_probs <= self.d)
        constrs.append(- self.z + self.m*self.state_probs <= self.d)
        constrs.append(cp.sum(self.d) <= self.beta * (1-self.m))

        objective = cp.Maximize(self.m)
        problem = cp.Problem(objective, constrs)
        problem.solve()
        print("----set-expansion solution----")
        print("m = {}, \n X(Dt)={}, \n abs_value_diff={}".format(self.m.value, self.z.value, self.d.value))

        # to finish: return a focus set

    def get_actions(self, states, cur_focus_set):
        pass

class SetOptPolicy(object):
    def __init__(self, sspa_size, y, N, act_frac, W):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.W = W
        self.W_sqrt = scipy.linalg.sqrtm(W)
        self.Lw = 2 * np.linalg.norm(W, ord=2) ## we add a small amount to ensure that the returned value is ...
        self.EPS = 1e-7
        # # the focus set, represented as an array of IDs of the arms
        # self.focus_set = np.array([])

        # variables and parameters
        self.z = cp.Variable(self.sspa_size)
        self.m = cp.Variable()
        self.d = cp.Variable(self.sspa_size)
        self.f = cp.Variable()
        self.s_count_scaled = cp.Parameter(self.sspa_size)
        self.beta = cp.Parameter()
        self.beta = min(act_frac, 1-act_frac)
        self.state_probs = cp.Parameter(self.sspa_size)

        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_new_focus_set(self, states):
        """
        states: length-N vector of states
        """
        s2indices = {s: None for s in self.sspa}
        s_count_scaled = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(states == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
        print("Xt([N]) = {}".format(s_count_scaled))
        self.s_count_scaled = s_count_scaled

        constrs = []
        # second order cone constraint: norm{sqrt(W) @ (z- m * mu)}_2 <= self.f
        constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
        constrs.append(self.z <= self.s_count_scaled)
        constrs.append(self.z >= 0)
        constrs.append(self.m == cp.sum(self.z))
        constrs.append(self.z - self.m*self.state_probs <= self.d)
        constrs.append(- self.z + self.m*self.state_probs <= self.d)
        constrs.append(cp.sum(self.d) <= self.beta * (1-self.m))

        objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
        problem = cp.Problem(objective, constrs)
        problem.solve()
        print("----set-optimization solution----")
        print("m = {}, \n X(Dt)={}, \n m*mu={} \n abs_value_diff={}, \n Hw={}\n ".format(
            self.m.value, self.z.value, self.m.value*self.state_probs, self.d.value, self.f.value))

        # to finish: return a focus set

    def get_actions(self, states, cur_focus_set):
        pass


# def states_to_scaled_state_counts(sspa_size, N, states):
#     scaled_state_counts = np.zeros((sspa_size,)) # 2 is the action space size
#     for i in range(len(states)):
#         s = int(states[i])
#         scaled_state_counts[s] += 1
#     return scaled_state_counts / N

def sa_list_to_freq(sspa_size, states, actions):
    """
    :param sspa_size: the size of the state space S
    :param states: the states of the arms, which is a length-N array
    :param actions: the actions of the arms, which is a length-N array
    :return: the state-action frequency of the input states and actions
    """
    assert len(states) == len(actions)
    sa_pair_freq = np.zeros((sspa_size, 2)) # 2 is the action space size
    for i in range(len(states)):
        s = int(states[i])
        a = int(actions[i])
        sa_pair_freq[s,a] += 1
    return sa_pair_freq / len(states)

def states_from_state_fracs(sspa_size, N, state_fracs):
    """
    :param sspa_size: the size of the state space
    :param N: the number of arms
    :param state_fracs: the state frequency
    :return: a length-N array of states, such that the ratio of each state is given by state_fracs
    """
    states = np.zeros((N,))
    for s in range(sspa_size):
        start_ind = int(N * np.sum(state_fracs[0:s]))
        end_ind = int(N * np.sum(state_fracs[0:(s+1)]))
        states[start_ind: end_ind] = s
    return states

def drift_array_to_rgb_array(drift):
    """
    converting the drift array to a color array based on the direction and magnitude
    :param drift: the drift map
    :return: the color map
    """
    upward_arrows = np.expand_dims(drift > 0, axis=2)
    downward_arrows = np.expand_dims(drift < 0, axis=2)
    blue = np.array([0,0,1]).reshape((1,1,3))
    red = np.array([1,0,0]).reshape((1,1,3))
    rgb_array = upward_arrows * blue + downward_arrows * red
    #print(rgb_array)
    rgb_array = rgb_array.reshape((-1,3))
    print(rgb_array.shape)
    return rgb_array
