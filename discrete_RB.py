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
        # trans_dict = {sa_pair: new_state_probs for sa_pair in sa_pairs}
        self.trans_dict = {sa_pair: trans_tensor[sa_pair[0], sa_pair[1], :] for sa_pair in self.sa_pairs}
        # trans_dict = {sa_pair: reward for sa_pair in sa_pairs}
        self.reward_dict = {sa_pair: reward_tensor[sa_pair[0], sa_pair[1]] for sa_pair in self.sa_pairs}
        # initialize the state of the arms at 0
        if init_states is not None:
            self.states = init_states.copy()
        else:
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
        self.DUALSTEP = 0.05 # the discretization step size of dual variable when solving for Whittle's index
        assert np.all(self.reward_tensor >= self.MINREWARD)
        assert np.all(self.reward_tensor <= self.MAXREWARD)

        # variables
        self.y = cp.Variable((self.sspa_size, 2))
        self.dualvar = cp.Parameter(name="dualvar")

        # store some data of the solution.
        # the values might be outdated. Do not access them unless immediately after solving the correct LP.
        self.opt_value = None
        self.avg_reward = None
        self.opt_subsidy = None
        self.value_func_relaxed = np.zeros((self.sspa_size,))
        self.q_func_relaxed = np.zeros((self.sspa_size, 2))

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
        basic_constrs.append(self.y >= 1e-8)
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
        self.opt_value = problem.solve()
        print("--------LP solved, solution as below-------")
        print("Optimal value ", self.opt_value)
        print("Optimal var")
        print(self.y.value)
        print("---------------------------")
        return (self.opt_value, self.y.value)

    def solve_LP_Priority(self):
        ##### THIS IMPLEMENTATION COULD HAVE A BUG!!! DUAL VARIABLE IS NOT RELIABLE. NEED TO REWRITE ALL THIS.
        objective = self.get_objective()
        constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        self.opt_value = problem.solve(verbose=False)
        # print("Optimal value ", opt_value)
        # print("Optimal var")
        # print(self.y.value)

        # hack value function from the dual variables. Later we should rewrite the dual problem explicitly
        self.avg_reward = constrs[-1].dual_value     # the sign is positive, DO NOT CHANGE IT
        for i in range(self.sspa_size):
            self.value_func_relaxed[i] = - constrs[i].dual_value   # the sign is negative, DO NOT CHANGE IT
        self.opt_subsidy = constrs[self.sspa_size].dual_value  # optimal subsidy for passive actions

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
        relaxed_objective = self.get_relaxed_objective()
        constrs = self.get_stationary_constraints() + self.get_basic_constraints()
        problem = cp.Problem(relaxed_objective, constrs)
        subsidy_values = np.arange(self.MINREWARD, self.MAXREWARD, self.DUALSTEP)
        passive_table = np.zeros((self.sspa_size, len(subsidy_values)))
        for i, subsidy in enumerate(subsidy_values):
            self.dualvar.value = subsidy
            problem.solve()
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
        # sorting from states with large index to small index
        wi2state_keys_sorted = sorted(wi2state.keys(), reverse=True)
        wi2state_sorted = {key: wi2state[key] for key in wi2state_keys_sorted}
        #print(wi2state_sorted)
        priority_list = []
        for approx_wi in wi2state_sorted:
            priority_list += wi2state_sorted[approx_wi]
        # print(priority_list)
        return priority_list, indexable


class PriorityPolicy(object):
    def __init__(self, sspa_size, priority_list, N, act_frac): #, N=None, budget=None, act_frac=None):
        """
        :param sspa_size: this is not needed
        :param priority_list: a list of states represented by numbers, from high priority to low priority
        """
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.priority_list = priority_list
        self.act_frac = act_frac
        self.N = N
        # if act_frac is not None:
        #     self.act_frac = act_frac
        # if (N is not None) and (budget is not None):
        #     self.N = N
        #     self.budget = budget
        # if (act_frac is None) and ((N is None) or (budget is None)):
        #     raise ValueError

    def get_actions(self, cur_states):
        ### DO NOT MODIFY THE INPUT "CUR_STATE"
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


class DirectRandomPolicy(object):
    """
    The policy that makes decisions only based on the current real states
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
                self.policy[state, 0] = 1.0
                self.policy[state, 1] = 0.0
        assert np.all(np.isclose(np.sum(self.policy, axis=1), 1.0, atol=1e-4)), \
            "policy definition wrong, the action probs do not sum up to 1, policy = {} ".format(self.policy)

    def get_actions(self, cur_states):
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

    # def get_sa_pair_fracs(self, cur_state_fracs):
    #     sa_pair_fracs = np.zeros((self.sspa_size, 2))
    #     rem_budget_normalize = self.act_frac
    #     for state in self.priority_list:
    #         frac_arms_this_state = cur_state_fracs[state]
    #         if rem_budget_normalize >= frac_arms_this_state:
    #             sa_pair_fracs[state, 1] = frac_arms_this_state
    #             sa_pair_fracs[state, 0] = 0.0
    #             rem_budget_normalize -= frac_arms_this_state
    #         else:
    #             sa_pair_fracs[state, 1] = rem_budget_normalize
    #             sa_pair_fracs[state, 0] = frac_arms_this_state - rem_budget_normalize
    #             rem_budget_normalize = 0
    #     assert rem_budget_normalize == 0.0, "something is wrong, whittles index should use up all the budget"
    #     return sa_pair_fracs


class SimuPolicy(object):
    def __init__(self, sspa_size, trans_tensor, reward_tensor, y, N, act_frac):
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

        # if init_states is not None:
        #     self.virtual_states = init_states
        # else:
        #     self.virtual_states = np.random.choice(self.sspa, self.N)
        self.virtual_states = np.random.choice(self.sspa, self.N)

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

    def get_actions(self, cur_states, tb_rule=None, tb_param=None):
        """
        :param cur_states: current state, actually not needed
        :return: actions, virtual_actions
        """
        # generate budget using randomized rounding
        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding
        # the current implementation does not need to read cur states to generate actions
        # generate virtual actions according to virtual states
        vs2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            vs2indices[state] = np.where(self.virtual_states == state)[0]
        # generate sample of virtual states using the policy
        virtual_actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            virtual_actions[vs2indices[state]] = np.random.choice(self.aspa, size=len(vs2indices[state]), p=self.policy[state])

        # modify virtual actions into real actions, consider budget constraints; break ties UNIFORMLY at random

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
                level_i_indices = np.where(priority_levels[i])[0]
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
        # print(prev_states, self.virtual_states)
        # print(actions, virtual_actions)
        # print(agree_mask)
        agree_indices = np.where(agree_mask)[0]
        # for those arms whose virtual states agree with real states, update them in the same way as the real states
        self.virtual_states[agree_indices] = cur_states[agree_indices]

        sa2indices = {sa_pair:[] for sa_pair in self.sa_pairs}
        for sa_pair in self.sa_pairs:
            # find out the indices of disagreement arms whose (virtual state, action) = sa_pair
            sa2indices[sa_pair] = np.where(np.all([self.virtual_states == sa_pair[0],
                                                   virtual_actions == sa_pair[1],
                                                   1 - agree_mask], axis=0))[0]
        for sa_pair in self.sa_pairs:
            cur_indices = sa2indices[sa_pair[0], sa_pair[1]]
            self.virtual_states[cur_indices] = np.random.choice(self.sspa, size=len(cur_indices),
                                                                p=self.trans_tensor[sa_pair[0], sa_pair[1]])


