"""
This file simulates the discrete_RB, and tests the performance of different policies?

Need to figure out how to
1 Input (in sparse form and dense form)
2 Store (in sparse form and dense form)
3 Process (solve the relaxed LP)
4 Simulate (run the simulation)
the small MDP of each arm.

Also, how to represent the overall RB?
Let us use the per-arm representation instead of the counting representation.
"""

import numpy as np
import cvxpy as cp


class RB(object):
    """
    state 0, 1, 2, 3
    action ...
    transition simulation: if (state, action) -> [0.5, 0.5, 0, 0], sample 1 or 2; if 2, ... better do the sampling in batches
                            transitions from 0: discrete(probs_of_0, num_samples=N, ); transitions from 1: ...
                            then we get an S by N matrix A. Get the next state using fancy indexing A[states, range(n)]
                            This is wasting samples....
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor, N):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = 2
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]
        self.N = N
        # map of state action to next states
        # trans_dict = {sa_pair: new_state_probs for sa_pair in sa_pairs}
        self.trans_dict = {sa_pair: trans_tensor[sa_pair[0], sa_pair[1], :] for sa_pair in self.sa_pairs}
        # trans_dict = {sa_pair: reward for sa_pair in sa_pairs}
        self.reward_dict = {sa_pair: reward_tensor[sa_pair[0], sa_pair[1]] for sa_pair in self.sa_pairs}
        # initialize the state of the arms at 0
        self.states = np.zeros((self.N,))

    def get_states(self):
        return self.states.copy()

    def step(self, actions):
        sa2indices = {sa_pair:None for sa_pair in self.sa_pairs} # each key is a list of indices in the range of self.N
        # find out the arms whose (state, action) = sa_pair
        for sa_pair in self.sa_pairs:
            sa2indices[sa_pair] = np.where(np.all([self.states == sa_pair[0], actions == sa_pair[1]], axis=0))[0]
        #print(sa2indices)
        #print([len(value) for key, value in sa2indices.items()])
        instant_reward = 0
        for sa_pair in self.sa_pairs:
            next_states = np.random.choice(self.sspa, len(sa2indices[sa_pair]), p=self.trans_dict[sa_pair])
            self.states[sa2indices[sa_pair]] = next_states
            instant_reward += self.reward_dict[sa_pair] * len(sa2indices[sa_pair])
        # we normalize it by the number of arms!
        instant_reward = instant_reward / self.N
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
    states are
    """
    def __init__(self, sspa_size, trans_tensor, reward_tensor):
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
        self.state_fracs = np.zeros((self.sspa_size,))
        self.state_fracs[0] = 1

    def step(self, sa_pair_fracs):
        """
        :param sa_pair_fracs:
        :return: update the state fracs and get rewards
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
    def __init__(self, sspa_size, trans_tensor, reward_tensor, act_frac):
        # problem parameters
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = 2
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.act_frac = act_frac
        # some constants
        self.EPS = 1e-7  # numbers smaller than this are regard as zero
        self.MINREWARD = 0 # a lower bound on the set of possible rewards
        self.MAXREWARD = 2 # an upper bound on the set of possible rewards
        self.DUALSTEP = 0.1 # the discretization step size of dual variable when solving for Whittle's index
        assert np.all(self.reward_tensor >= self.MINREWARD)
        assert np.all(self.reward_tensor <= self.MAXREWARD)

        # variables
        self.y = cp.Variable((self.sspa_size, 2))
        self.dualvar = cp.Parameter(name="dualvar")

    def get_stationary_constraints(self):
        stationary_constrs = []
        for cur_s in self.sspa:
            # for sa_pair in self.sa_pairs:
            #     m_s += self.y[sa_pair[0], sa_pair[1]] * self.trans_tensor[sa_pair[0], sa_pair[1], cur_s]
            m_s = cp.sum(cp.multiply(self.y, self.trans_tensor[:,:,cur_s]))
            stationary_constrs.append(m_s == cp.sum(self.y[cur_s, :]))
        return stationary_constrs

    def get_budget_constraints(self):
        budget_constrs = []
        budget_constrs.append(cp.sum(self.y[:,1]) == self.act_frac)
        return budget_constrs

    def get_basic_constraints(self):
        basic_constrs = []
        basic_constrs.append(self.y >= 0)
        basic_constrs.append(cp.sum(self.y) == 1)
        return basic_constrs

    def get_objective(self):
        objective = cp.Maximize(cp.sum(cp.multiply(self.y, self.reward_tensor)))
        return objective

    def get_relaxed_objective(self):
        subsidy_reward_1 = self.reward_tensor[:,1]
        subsidy_reward_0 = self.reward_tensor[:,0] + self.dualvar
        relaxed_objective = cp.Maximize(cp.sum(cp.multiply(self.y[:,1], subsidy_reward_1))
                                        + cp.sum(cp.multiply(self.y[:,0], subsidy_reward_0)))
        return relaxed_objective

    def solve_lp(self):
        objective = self.get_objective()
        constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        opt_value = problem.solve()
        print("--------LP solved, solution as below-------")
        print("Optimal value ", opt_value)
        print("Optimal var")
        print(self.y.value)
        print("---------------------------")
        return (opt_value, self.y.value)

    def y2randomized_policy(self, y_values):
        pass

    def y2waterfilling_policy(self, y_values):
        for state in self.sspa:
            if (y_values[state, 0] > self.EPS) and (self.y.value[state, 1] < self.EPS):
                # fluid passive states
                pass
            elif (y_values[state, 0] < self.EPS) and (self.y.value[state, 1] > self.EPS):
                # fluid active states
                # calculate the action gap...
                pass
            elif (y_values[state, 0] > self.EPS) and (self.y.value[state, 1] > self.EPS):
                # fluid neutral states
                pass
            else:
                # fluid null states
                pass

    def solve_whittles_policy(self):
        relaxed_objective = self.get_relaxed_objective()
        constrs = self.get_stationary_constraints() + self.get_basic_constraints()
        problem = cp.Problem(relaxed_objective, constrs)
        subsidy_values = np.arange(self.MINREWARD, self.MAXREWARD, self.DUALSTEP)
        passive_table = np.zeros((self.sspa_size, len(subsidy_values)))
        for i, subsidy in enumerate(subsidy_values):
            self.dualvar.value = subsidy
            problem.solve()
            #print(self.y.value)
            for state in self.sspa:
                if (self.y.value[state, 0] > self.EPS) and (self.y.value[state, 1] < self.EPS):
                    passive_table[state, i] = 1
        wi2state = {}
        for state in self.sspa:
            approx_wi = np.where(passive_table[state, :])[0][0]  # find the smallest subsidy such that state becomes passive
            indexable = np.all(passive_table[state, approx_wi:] == 1) #, "not indexable"  # check indexability
            if approx_wi not in wi2state:
                wi2state[approx_wi] = [state]
            else:
                print("Warning: two states have the same Whittle's index. Breaking ties favoring smaller states")
                wi2state[approx_wi].append(state)
        # sort from larger indices to smaller indices
        sorted(wi2state)
        # print(wi2state)
        priority_list = []
        for approx_wi in wi2state:
            priority_list += wi2state[approx_wi]
        # print(priority_list)
        return priority_list, indexable


class PriorityPolicy(object):
    def __init__(self, sspa_size, priority_list, N=None, budget=None, act_frac=None):
        """
        :param sspa_size: this is not needed
        :param priority_list: a list of states represented by numbers, from high priority to low priority
        """
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.priority_list = priority_list
        if act_frac is not None:
            self.act_frac = act_frac
        if (N is not None) and (budget is not None):
            self.N = N
            self.budget = budget
        if (act_frac is None) and ((N is None) or (budget is None)):
            raise ValueError

    def get_actions(self, cur_states):
        ### DO NOT MODIFY THE INPUT "CUR_STATE"
        # return actions from states
        s2indices = {state:None for state in self.sspa} # each key is a list of indices in the range of self.N
        # find out the arms whose (state, action) = sa_pair
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]

        actions = np.zeros((self.N,))
        rem_budget = self.budget
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


class WaterfillingPolicy(object):
    def __init__(self, sspa_size, y, N, budget):
        pass

    def get_actions(self, cur_state):
        pass


class SimuPolicy(object):
    def __init__(self, sspa_size, trans_tensor, reward_tensor, N, budget, y):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.N = N
        self.budget = budget
        self.y = y
        self.EPS = 1e-7

        self.virtual_states = np.random.choice(self.sspa, self.N)
        #self.virtual_states = initial_states

        # get the randomized policy from the solution y
        self.state_probs = np.sum(self.y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = self.y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 1.0
                self.policy[state, 1] = 0.0
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_actions(self, cur_states):
        # the current implementation does not need to read cur states to generate actions
        # generate virtual actions according to virtual states
        vs2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            vs2indices[state] = np.where(self.virtual_states == state)[0]
        # generate sample of virtual states using the policy
        actions = np.zeros((self.N,))
        for state in self.sspa:
            actions[vs2indices[state]] = np.random.choice(self.aspa, size=len(vs2indices[state]), p=self.policy[state])

        # modify virtual actions into real actions, consider budget constraints; break ties UNIFORMLY at random
        num_requests = np.sum(actions)
        if num_requests > self.budget:
            indices_request = np.where(actions==1)[0]
            request_ignored = np.random.choice(indices_request, int(num_requests - self.budget), replace=False)
            actions[request_ignored] = 0
        else:
            indices_no_request = np.where(actions==0)[0]
            no_request_pulled = np.random.choice(indices_no_request, int(self.budget - num_requests), replace=False)
            actions[no_request_pulled] = 1

        return actions

    def virtual_step(self, prev_states, cur_states, actions):
        # simulation with coupling
        agree_mask = self.virtual_states == prev_states
        agree_indices = np.where(agree_mask)[0]
        # for those arms whose virtual states agree with real states, update them in the same way as the real states
        self.virtual_states[agree_indices] = cur_states[agree_indices]

        sa2indices = {sa_pair:[] for sa_pair in self.sa_pairs}
        for sa_pair in self.sa_pairs:
            # find out the indices of disagreement arms whose (virtual state, action) = sa_pair
            sa2indices[sa_pair] = np.where(np.all([self.virtual_states == sa_pair[0],
                                                   actions == sa_pair[1],
                                                   1-agree_mask], axis=0))[0]
        for sa_pair in self.sa_pairs:
            cur_indices = sa2indices[sa_pair[0], sa_pair[1]]
            self.virtual_states[cur_indices] = np.random.choice(self.sspa, size=len(cur_indices),
                                                                p=self.trans_tensor[sa_pair[0], sa_pair[1]])


