"""
This file defines the class for discrete_RB simulation environment,
the helper class for solving single-armed LPs and getting priorities,
classes for RB policies,
along with a few helper functions
"""

import numpy as np
import cvxpy as cp
import scipy
import warnings


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
        # min_reward = np.min(self.reward_tensor)
        # max_reward = np.max(self.reward_tensor)
        # self.MINPENALTY = - 5*(max_reward - min_reward)   # a lower bound on the range of penalty for active actions
        # self.MAXPENALTY = 5*(max_reward - min_reward) # an upper bound on the range of possible penalty for active ations
        # self.DUALSTEP = (max_reward - min_reward) / 10 # 0.05 # the discretization step size of dual variable when solving for Whittle's index

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

        self.policy = None
        self.state_probs = None
        self.Ppibs = None

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

    def solve_lp(self, verbose=True):
        """
        run this before doing anything
        """
        objective = self.get_objective()
        constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        self.opt_value = problem.solve()
        if verbose:
            print("--------LP relaxation solved, solution as below-------")
            print("Optimal value ", self.opt_value)
            print("Optimal var")
            print(self.y.value)

        # compute some essential quantities and for later use
        y = self.y.value
        ## make sure y is non-negative and sums up to 1
        y = y * (y>=0)
        y = y / np.sum(y)
        self.state_probs = np.sum(y, axis=1)
        self.policy = np.zeros((self.sspa_size, 2)) # conditional probability of actions given state
        for state in self.sspa:
            if self.state_probs[state] > self.EPS:
                self.policy[state, :] = y[state, :] / self.state_probs[state]
            else:
                self.policy[state, 0] = 0.5
                self.policy[state, 1] = 0.5
        if verbose:
            print("mu*=", self.state_probs)
            print("pibs policy=", self.policy)
            print("---------------------------")
        self.Ppibs = np.zeros((self.sspa_size, self.sspa_size,))
        for a in self.aspa:
            self.Ppibs += self.trans_tensor[:,a,:]*np.expand_dims(self.policy[:,a], axis=1)

        return (self.opt_value, y)

    def solve_LP_Priority(self, fixed_dual=None):
        if fixed_dual is None:
            objective = self.get_objective()
            constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        else:
            self.dualvar.value = fixed_dual
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

        print("---solving LP Priority----")
        print("lambda* = ", self.opt_subsidy)
        print("avg_reward = ", self.avg_reward)
        print("value_func = ", self.value_func_relaxed)

        for i in range(self.sspa_size):
            for j in range(2):
                self.q_func_relaxed[i, j] = self.reward_tensor[i, j] + self.opt_subsidy * (j==0) - self.avg_reward + np.sum(self.trans_tensor[i, j, :] * self.value_func_relaxed)
        print("q func = ", self.q_func_relaxed)
        print("action gap =  ", self.q_func_relaxed[:,1] - self.q_func_relaxed[:,0])
        print("---------------------------")

        priority_list = np.flip(np.argsort(self.q_func_relaxed[:,1] - self.q_func_relaxed[:,0]))
        return list(priority_list)

    def understand_lagrange_relaxation(self, lower, upper, stepsize):
        # solve a set of relaxed problem with different subsidy values, and find out the Whittle's index
        relaxed_objective = self.get_relaxed_objective()
        constrs = self.get_stationary_constraints() + self.get_basic_constraints()
        problem = cp.Problem(relaxed_objective, constrs)
        subsidy_values = np.arange(lower, upper, stepsize)
        state_type_table = np.ones((self.sspa_size, len(subsidy_values))) # each row is a state, each column is a dual value
        for i, subsidy in enumerate(subsidy_values):
            self.dualvar.value = subsidy
            problem.solve(abstol=1e-13)
            for state in self.sspa:
                if (self.y.value[state, 0] > self.EPS) and (self.y.value[state, 1] < self.EPS):
                    state_type_table[state, i] = 0
                elif (self.y.value[state, 0] <= self.EPS) and (self.y.value[state, 1] < self.EPS):
                    state_type_table[state, i] = 2
            y = self.y.value * (self.y.value > self.EPS)
            print(y.T)
        print(state_type_table)


    def solve_whittles_policy(self):
        """
        return -1 if non-indexable
        return -2 if multichain
        otherwise return the whittle indices
        """
        ## iterates: mu^k_i, alpha^k_i, pi^k
        ## they are computed in the order: mu^{k-1}_i, pi^k -> alpha^k_i(mu^{k-1}_min) -> mu^k_i -> pi^{k+1}
        ## mu_table[k, i] = mu^k_i, pi_table[k,i] = pi^k_i
        mu_table = np.infty * np.ones((self.sspa_size, self.sspa_size))
        whittle_indices = np.nan * np.ones((self.sspa_size,))
        ## a deterministic single-armed policy for the lagrange relaxed problem; entries denote probability of activation
        pi = np.ones((self.sspa_size,))
        ## two useful quantities that are fixed over iterations
        delta = self.reward_tensor[:,1] - self.reward_tensor[:,0]
        Delta = self.trans_tensor[:,1,:] - self.trans_tensor[:,0,:]
        Delta[:,0] = 0
        for k in range(self.sspa_size):
            if k == 0:
                prev_mu_min = -np.infty #self.MINPENALTY
                alpha_mu_prev = np.infty * np.ones((self.sspa_size,))
            else:
                prev_mu_min = cur_mu_min #np.min(mu_table[(k-1),:])
                ## note that alpha^{pi^k}(mu^{k-1}) = alpha^{pi^{k-1}}(mu^{k-1})
                alpha_mu_prev = alpha_mu_k
            ## evaluate matrix A_pi for the current policy pi^k to test unichain
            r_pi = self.reward_tensor[:,1] * pi + self.reward_tensor[:,0] * (1-pi)
            P_pi = self.trans_tensor[:,1,:] * np.expand_dims(pi, axis=1) +  self.trans_tensor[:,0,:] * np.expand_dims((1-pi), axis=1)
            A_pi = np.eye(self.sspa_size)
            A_pi[:,0] = 1
            P_pi_modified = P_pi.copy()
            P_pi_modified[:,0] = 0
            A_pi = A_pi - P_pi_modified
            if np.linalg.cond(A_pi) > 1e8:
                print("pi={}, np.linalg.cond(A_pi)={}".format(pi, np.linalg.cond(A_pi)))
                return -2
            # print("pi={}, lam={}, alpha mu prev={}".format(pi, prev_mu_min, alpha_mu_prev))
            ## compute mu^k
            d_pi = np.linalg.solve(A_pi, -pi)
            v_pi_pure = np.linalg.solve(A_pi, r_pi)
            for i in range(self.sspa_size):
                if pi[i] == 0:
                    continue
                else:
                    # assert alpha_pi[i] >= -self.EPS, "alpha_pi[{}]={}<0, but pi[{}]={}>0".format(i, alpha_pi[i], i, pi[i])
                    if alpha_mu_prev[i] == 0:
                        mu_table[k, i] = prev_mu_min
                    else:
                        if 1 - np.matmul(Delta[i,:].T, d_pi) > 0:
                            # mu_table[k, i] = prev_mu_min +  alpha_pi[i] / (1-np.matmul(Delta[i,:].T, d_pi))
                            mu_table[k, i] = (delta[i] + np.matmul(Delta[i,:].T, v_pi_pure)) / (1-np.matmul(Delta[i,:].T, d_pi))
                        else:
                            mu_table[k, i] = np.infty
            cur_mu_min = np.min(mu_table[k,:])
            # alpha_mu_k = alpha_pi  - (cur_mu_min - prev_mu_min) * (1 - np.matmul(Delta, d_pi))
            ## compute alpha^{pi_k}(mu^k)
            v_pi = np.linalg.solve(A_pi, r_pi - cur_mu_min*pi)
            alpha_mu_k = delta - cur_mu_min*np.ones((self.sspa_size,)) + np.matmul(Delta, v_pi)
            ## test indexability
            if (prev_mu_min < cur_mu_min - self.EPS) and np.any((alpha_mu_k>=self.EPS)*(1-pi)):
                # print(cur_mu_min)
                # print(pi)
                # print(alpha_mu_k)
                return -1
            ## return if cur_mu_min reach infty
            if cur_mu_min == np.infty:
                whittle_indices[pi] = np.infty
                return whittle_indices
            ## update policy pi from pi^k to pi^{k+1}
            deactivate_states = np.where(np.all([alpha_mu_k < self.EPS, alpha_mu_k > -self.EPS], axis=0))[0]
            whittle_indices[deactivate_states] = cur_mu_min
            pi[deactivate_states] = 0
        return np.flip(np.argsort(whittle_indices))

    def compute_W(self, abstol):
        Ppibs_centered = self.Ppibs - np.outer(np.ones((self.sspa_size,)), self.state_probs)
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

    def get_future_expected_budget_requirements(self, state_dist, T_ahead):
        cur_state_dist = state_dist
        budget_reqs = []
        for t in range(T_ahead):
            budget_reqs.append(np.matmul(self.policy[:,1].T, cur_state_dist))
            cur_state_dist = np.matmul(self.Ppibs.T, cur_state_dist)
        return budget_reqs

    def get_future_budget_req_bounds_Lone(self, state_dist, T_ahead):
        beta = min(self.act_frac, 1-self.act_frac)
        cur_state_dist = state_dist
        upper_bounds = []
        lower_bounds = []
        for t in range(T_ahead):
            cur_Lone_norm = np.linalg.norm(cur_state_dist - self.state_probs, ord=1)
            upper_bounds.append(self.act_frac + beta * cur_Lone_norm)
            lower_bounds.append(self.act_frac - beta * cur_Lone_norm)
            cur_state_dist = np.matmul(self.Ppibs.T, cur_state_dist)
        return upper_bounds, lower_bounds

    def get_future_budget_req_bounds_Wnorm(self, state_dist, T_ahead):
        beta = min(self.act_frac, 1-self.act_frac)
        W = self.compute_W(abstol=1e-10)[0]
        cpibs = self.policy[:,1]
        ratiocw = np.sqrt(np.matmul(cpibs.T, np.linalg.solve(W, cpibs)))
        cur_state_dist = state_dist
        upper_bounds = []
        lower_bounds = []
        for t in range(T_ahead):
            x_minus_mu = cur_state_dist - self.state_probs
            cur_W_norm = np.sqrt(np.matmul(np.matmul(x_minus_mu.T, W), x_minus_mu))
            upper_bounds.append(self.act_frac + beta / ratiocw * cur_W_norm)
            lower_bounds.append(self.act_frac - beta / ratiocw * cur_W_norm)
            cur_state_dist = np.matmul(self.Ppibs.T, cur_state_dist)
        return upper_bounds, lower_bounds

    def compute_Phi(self, verbose=True):
        y = self.y.value
        ind_neu = np.where(np.all([y[:,0] > self.EPS, y[:,1] > self.EPS],axis=0))[0]
        if verbose:
            print("------computing Phi-------")
            print("ind_neu=", ind_neu)
        if len(ind_neu) == 0:
            Phi = self.Ppibs -  np.outer(np.ones((self.sspa_size,)), self.state_probs)
        else:
            if len(ind_neu) > 1:
                warnings.warn("Multiple neutral state; using the first one for computing Phi")
                ind_neu = ind_neu[0:1]
            Phi = self.Ppibs -  np.outer(np.ones((self.sspa_size,)), self.state_probs) \
                - np.outer(self.policy[:,1] - self.act_frac * np.ones((self.sspa_size,)), self.trans_tensor[ind_neu,1,:] - self.trans_tensor[ind_neu,0,:])
        moduli = [np.absolute(lam) for lam in np.linalg.eigvals(Phi)]
        spec_rad = max(moduli)
        # print the result
        # print("P1=", self.trans_tensor[:,1,:])
        # print("P0=", self.trans_tensor[:,0,:])
        # print("Ppibs=", self.Ppibs)
        if verbose:
            print("Phi=", Phi)
            print("moduli of Phi's eigenvalues=", moduli)
            print("spectral radius of Phi=", spec_rad)
            print("---------------------------")
        return Phi

    def compute_U(self, abstol):
        Phi = self.compute_Phi()
        if np.max(np.abs(np.linalg.eigvals(Phi))) >= 1:
            return np.infty, np.infty

        Phi_power = np.eye(self.sspa_size)
        U = np.zeros((self.sspa_size, self.sspa_size))
        # start calculating
        iters = 0
        while True:
            U += np.matmul(Phi_power, Phi_power.T)
            Phi_power = np.matmul(Phi, Phi_power)
            Phi_power_norm = np.linalg.norm(Phi_power)
            U_norm = np.linalg.norm(U, ord=2)
            # print("U norm = {}, Phi_power_norm = {}".format(U_norm, Phi_power_norm))
            spn_error = U_norm * Phi_power_norm**2 / (1-Phi_power_norm**2)
            iters += 1
            if (Phi_power_norm < 1) and (spn_error < abstol):
                break
        print("U computed after expanding {} terms, error={}".format(iters, spn_error))
        return U, spn_error




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

        if init_virtual is None:
            self.virtual_states = np.random.choice(self.sspa, self.N, replace=True, p=self.state_probs)
        else:
            self.virtual_states = init_virtual.copy()

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
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding when alpha N

        num_requests = np.sum(actions)
        if num_requests > budget:
            indices_request = np.sort(np.where(actions==1)[0])
            request_ignored = indices_request[(-int(num_requests - budget)):]
            actions[request_ignored] = 0
            num_action_rect = len(request_ignored)
        elif num_requests < budget:
            indices_no_request = np.sort(np.where(actions==0)[0])
            no_request_pulled = indices_no_request[(-int(budget - num_requests)):]
            actions[no_request_pulled] = 1
            num_action_rect = len(no_request_pulled)
        else:
            num_action_rect = 0
        assert np.sum(actions) == budget

        return actions, self.N - num_action_rect


class SetExpansionPolicy(object):
    def __init__(self, sspa_size, y, N, act_frac, W=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        if W is not None:
            self.W = W
            self.W_sqrt = scipy.linalg.sqrtm(W)
        else:
            self.W = None
            self.W_sqrt = None
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
        self.beta.value = min(act_frac, 1-act_frac)

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

        if W is not None:
            self.cpibs = self.policy[:,1]
            self.ratiocw = np.sqrt(np.matmul(self.cpibs.T, np.linalg.solve(self.W, self.cpibs)))
            print("W=", self.W)
            print("cpibs=", self.cpibs)
            print("ratiocw=", self.ratiocw)

    def get_new_focus_set(self, cur_states, last_focus_set, subproblem="L1"):
        """
        states: length-N vector of states
        focus_set: array of IDs for the arms in the focus set
        """
        if len(last_focus_set) > 0:
            states_fs = cur_states[last_focus_set]
        else:
            states_fs = np.array([])
        s2indices = {s: None for s in self.sspa}
        s2indices_fs = {s: None for s in self.sspa} # state to indices map in the focus set
        s_count_scaled = np.zeros((self.sspa_size,))
        s_count_scaled_fs = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(cur_states == s)[0]
            s2indices_fs[s] = np.where(states_fs == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
            s_count_scaled_fs[s] = len(s2indices_fs[s]) / self.N
        # print("Xt([N]) = {}, Xt(D(t-1)) = {}".format(s_count_scaled, s_count_scaled_fs))
        self.s_count_scaled.value = s_count_scaled
        self.s_count_scaled_fs.value = s_count_scaled_fs

        if subproblem == "L1":
            cur_m = len(last_focus_set)/self.N
            ## 0617 update: multiply 0.5 in the L1 norm
            cur_delta = self.beta.value*(1-cur_m) - 0.5*np.linalg.norm(s_count_scaled_fs - cur_m * self.state_probs, ord=1)
            constrs = []
            non_shrink_flag = 0
            if cur_delta >= 0:
                constrs.append(self.z <= self.s_count_scaled)
                constrs.append(self.z >= self.s_count_scaled_fs)
                non_shrink_flag = 1
            else:
                constrs.append(self.z <= self.s_count_scaled_fs)
                constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(self.z - self.m*self.state_probs <= self.d)
            constrs.append(- self.z + self.m*self.state_probs <= self.d)
            ## 0617 update: multiply 0.5 in the L1 norm
            constrs.append(0.5*cp.sum(self.d) <= self.beta * (1-self.m))
            objective = cp.Maximize(self.m)
            problem = cp.Problem(objective, constrs)
            problem.solve()
            # print("----set-expansion solution----")
            # print("m = {}, \n X(Dt)={}, \n abs_value_diff={}".format(self.m.value, self.z.value, self.d.value))
        elif subproblem == "W":
            cur_m = len(last_focus_set)/self.N
            x_minus_mmu = s_count_scaled_fs - cur_m * self.state_probs
            ## todo: check if it also needs to multiply by 2 here
            cur_delta = self.beta.value*(1-cur_m) - self.ratiocw * np.sqrt(np.matmul(np.matmul(x_minus_mmu.T, self.W), x_minus_mmu))
            constrs = []
            non_shrink_flag = 0
            if cur_delta > 0:
                constrs.append(self.z <= self.s_count_scaled)
                constrs.append(self.z >= self.s_count_scaled_fs)
                non_shrink_flag = 1
            else:
                constrs.append(self.z <= self.s_count_scaled_fs)
                constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(cp.SOC(self.beta*(1-self.m)/self.ratiocw, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            objective = cp.Maximize(self.m)
            problem = cp.Problem(objective, constrs)
            problem.solve()
            # print("----set-expansion solution----")
            # print("m = {}, \n X(Dt)={}, \n abs_value_diff={}".format(self.m.value, self.z.value, self.d.value))
        else:
            raise NotImplementedError

        next_focus_set = []
        for s in self.sspa:
            next_s_count_fs = int(self.N * self.z.value[s])
            next_focus_set.extend(s2indices[s][0:next_s_count_fs])
        next_focus_set = np.array(next_focus_set, dtype=int)
        # print("state count = ", self.z.value * self.N, "states = ", cur_states, "focus set = ", next_focus_set)
        return next_focus_set, non_shrink_flag

    def get_actions(self, cur_states, cur_focus_set, tb_rule="random", tb_priority=None):
        """
        :param cur_states: the current states of the arms
        :param cur_focus_set: array of IDs denoting the arms in the focus set
        :param tb_rule: random, ID, priority, which defines the tie breaking within and out of the focus set
        :param rb_priority: np.array, defining the priority of states if tb_rule == priority, range from 1,2,3..|S|,
                            smaller number means higher priority
        """
        # make sure priority >=1, for the convenience of later operations
        if tb_rule == "priority":
            tb_priority = np.array(tb_priority)
            tb_priority += 1
            assert np.all(tb_priority >= 1)

        s2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]
        actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])

        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding when alpha N

        cur_focus_set_mask = np.zeros((self.N,), dtype=int)
        cur_focus_set_mask[cur_focus_set] = 1
        # print("cur_focus_set = {}, mask = {}", cur_focus_set, cur_focus_set_mask)
        req_focus_mask = actions * cur_focus_set_mask
        non_req_focus_mask = (1-actions) * cur_focus_set_mask
        non_focus = np.where(1-cur_focus_set_mask)[0]
        # print("actions = {}\nfocus_set_mask = {}\nreq_focus_mask = {}".format(actions, cur_focus_set_mask, req_focus_mask))

        num_req = np.sum(actions)
        num_req_fs = np.sum(req_focus_mask)
        num_non_req_fs = np.sum(non_req_focus_mask)
        conformity_flag = 0
        if num_req_fs >= budget:
            # rectify focus set by setting some of the actions from one to zero; set other actions to zero
            req_focus = np.where(req_focus_mask)[0]
            if tb_rule == "random":
                np.random.shuffle(req_focus)
                actions[req_focus[budget:]] = 0
            elif tb_rule == "ID":
                req_focus = np.sort(req_focus)
                actions[req_focus[budget:]] = 0
            elif tb_rule == "priority":
                # find priority of arms using tb_priority
                arm_priorities = tb_priority[cur_states]
                # focus on the arms outside focus set and requesting pulls
                arm_priorities = arm_priorities * req_focus_mask
                sorted_indices = np.argsort(arm_priorities)
                actions[sorted_indices[(self.N-(num_req_fs-budget)):]] = 0 ####
            else:
                raise NotImplementedError
            actions[non_focus] = 0
        elif num_non_req_fs >= (self.N - budget):
            # rectify focus set by setting some of the actions from one to zero; set other actions to one
            non_req_focus = np.where(non_req_focus_mask)[0]
            if tb_rule == "random":
                np.random.shuffle(non_req_focus)
                actions[non_req_focus[(self.N - budget):]] = 1
            elif tb_rule == "ID":
                non_req_focus = np.sort(non_req_focus)
                actions[non_req_focus[(self.N - budget):]] = 1
            elif tb_rule == "priority":
                # find priority of arms using tb_priority
                arm_priorities = tb_priority[cur_states]
                # focus on the arms outside focus set and requesting pulls
                arm_priorities = arm_priorities * non_req_focus_mask
                sorted_indices = np.argsort(arm_priorities)
                actions[sorted_indices[self.N-(num_non_req_fs+budget-self.N):]] = 1
                # print(num_non_req_fs - (self.N - budget), len(sorted_indices[-(num_non_req_fs+budget-self.N):]))
            else:
                raise NotImplementedError
            actions[non_focus] = 1
        else:
            # no rectify focus set; take suitable number of actions to zero and one
            # print("case 3: actions before rect=", actions)
            ## tie-breaking based on ID
            if num_req > budget:
                req_non_focus_mask = actions * (1-cur_focus_set_mask)
                req_non_focus = np.where(req_non_focus_mask)[0]
                if tb_rule == "random":
                    np.random.shuffle(req_non_focus)
                    actions[req_non_focus[(budget-num_req_fs):]] = 0
                elif tb_rule == "ID":
                    req_non_focus = np.sort(req_non_focus)
                    actions[req_non_focus[(budget-num_req_fs):]] = 0
                elif tb_rule == "priority":
                    # find priority of arms using tb_priority
                    arm_priorities = tb_priority[cur_states]
                    # focus on the arms outside focus set and requesting pulls
                    arm_priorities = arm_priorities * req_non_focus_mask
                    sorted_indices = np.argsort(arm_priorities)
                    actions[sorted_indices[-(num_req-budget):]] = 0
                else:
                    raise NotImplementedError
            elif budget > num_req:
                non_req_non_focus_mask = (1-actions) * (1-cur_focus_set_mask)
                non_req_non_focus = np.where(non_req_non_focus_mask)[0]
                if tb_rule == "random":
                    np.random.shuffle(non_req_non_focus)
                    actions[non_req_non_focus[(self.N-budget-num_non_req_fs):]] = 1
                elif tb_rule == "ID":
                    req_non_focus = np.sort(non_req_non_focus)
                    actions[non_req_non_focus[(self.N-budget-num_non_req_fs):]] = 1
                elif tb_rule == "priority":
                    # find priority of arms using tb_priority
                    arm_priorities = tb_priority[cur_states]
                    # focus on the arms outside focus set and requesting pulls
                    arm_priorities = arm_priorities * non_req_non_focus_mask
                    sorted_indices = np.argsort(arm_priorities)
                    actions[sorted_indices[-(budget-num_req):]] = 1 ####
                else:
                    raise NotImplementedError
            else:
                pass
            conformity_flag = 1
        assert np.sum(actions) == budget, "np.sum(actions)={}, budget={}".format(np.sum(actions), budget)

        return actions, conformity_flag


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
        self.Lw = 2 * np.linalg.norm(W, ord=2)
        self.EPS = 1e-7
        # # the focus set, represented as an array of IDs of the arms
        # self.focus_set = np.array([])

        # variables and parameters
        self.z = cp.Variable(self.sspa_size)
        self.m = cp.Variable()
        self.d = cp.Variable(self.sspa_size)
        self.f = cp.Variable()
        self.lam_up = cp.Variable()
        self.lam_low = cp.Variable()
        self.gamma_up = cp.Variable(self.sspa_size)
        self.gamma_low = cp.Variable(self.sspa_size)
        self.s_count_scaled = cp.Parameter(self.sspa_size)
        self.beta = cp.Parameter()
        self.beta.value = min(act_frac, 1-act_frac)

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

        self.cpibs = self.policy[:,1]
        self.ratiocw = np.sqrt(np.matmul(self.cpibs.T, np.linalg.solve(self.W, self.cpibs)))
        # print("W=", self.W)
        # print("cpibs=", self.cpibs)
        # print("ratiocw=", self.ratiocw)

    def get_new_focus_set(self, cur_states, subproblem="L1"):
        """
        cur_states: length-N vector of states
        """
        s2indices = {s: None for s in self.sspa}
        s_count_scaled = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(cur_states == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
        # print("Xt([N]) = {}".format(s_count_scaled))
        self.s_count_scaled.value = s_count_scaled

        if subproblem == "L1":
            constrs = []
            # second order cone constraint: norm{sqrt(W) @ (z- m * mu)}_2 <= self.f
            constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(self.z - self.m*self.state_probs <= self.d)
            constrs.append(- self.z + self.m*self.state_probs <= self.d)
            ## 0617 update: multiply 0.5 in front of the L1 norm
            constrs.append(0.5*cp.sum(self.d) <= self.beta * (1-self.m))
            objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
            problem = cp.Problem(objective, constrs)
            problem.solve()
            # print("Lyapunov value = ", problem.value)
            # print("----set-optimization solution----")
            # print("m = {}, \n X(Dt)={}, \n m*mu={} \n abs_value_diff={}, \n Hw={}\n ".format(
            #     self.m.value, self.z.value, self.m.value*self.state_probs, self.d.value, self.f.value))
        elif subproblem == "W":
            constrs = []
            ## todo: check if it also needs to multiply by 0.5 here
            constrs.append(cp.SOC(self.beta*(1-self.m)/self.ratiocw, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            objective = cp.Minimize(1 - self.m)
            problem = cp.Problem(objective, constrs)
            problem.solve()
        elif subproblem == "c":
            constrs = []
            constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(np.matmul(self.cpibs.T, self.z) <= self.act_frac)
            constrs.append(np.matmul(self.cpibs.T, self.z) >= self.act_frac - (1-self.m))
            objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
            problem = cp.Problem(objective, constrs)
            problem.solve()
        elif subproblem == "tight":
            constrs = []
            # second order cone constraint: norm{sqrt(W) @ (z- m * mu)}_2 <= self.f
            constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            # future budget requirement upper bound
            constrs.append(self.act_frac*self.lam_up + cp.sum(self.gamma_up) <= self.act_frac)
            constrs.append(self.state_probs*self.lam_up + self.gamma_up >= self.z)
            constrs.append(self.gamma_up >= 0)
            # future budget requirement lower bound
            constrs.append(self.act_frac*self.lam_low + cp.sum(self.gamma_low) + (1-self.m) >= self.act_frac)
            constrs.append(self.state_probs*self.lam_low + self.gamma_low <= self.z)
            constrs.append(self.gamma_low >= 0)
            # objective
            objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
            problem = cp.Problem(objective, constrs)
            problem.solve()
        else:
            raise NotImplementedError

        next_focus_set = []
        for s in self.sspa:
            next_s_count_fs = int(self.N * self.z.value[s])
            next_focus_set.extend(s2indices[s][0:next_s_count_fs])
        next_focus_set = np.array(next_focus_set, dtype=int)
        return next_focus_set

    def get_actions(self, cur_states, cur_focus_set, tb_rule="random", tb_priority=None):
        """
        :param cur_states: the current states of the arms
        :param cur_focus_set: array of IDs denoting the arms in the focus set
        :param tb_rule: random, ID, priority, which defines the tie breaking within and out of the focus set
        :param rb_priority: np.array, defining the priority of states if tb_rule == priority, range from 1,2,3..|S|,
                            smaller number means higher priority
        """
        # make sure priority >=1, for the convenience of later operations
        if tb_rule == "priority":
            tb_priority = np.array(tb_priority)
            tb_priority += 1
            assert np.all(tb_priority >= 1)

        s2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]
        actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])

        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding when alpha N

        cur_focus_set_mask = np.zeros((self.N,), dtype=int)
        cur_focus_set_mask[cur_focus_set] = 1
        # print("cur_focus_set = {}, mask = {}", cur_focus_set, cur_focus_set_mask)
        req_focus_mask = actions * cur_focus_set_mask
        non_req_focus_mask = (1-actions) * cur_focus_set_mask
        non_focus = np.where(1-cur_focus_set_mask)[0]
        # print("actions = {}\nfocus_set_mask = {}\nreq_focus_mask = {}".format(actions, cur_focus_set_mask, req_focus_mask))

        num_req = np.sum(actions)
        num_req_fs = np.sum(req_focus_mask)
        num_non_req_fs = np.sum(non_req_focus_mask)
        conformity_flag = 0
        if num_req_fs >= budget:
            # rectify focus set by setting some of the actions from one to zero; set other actions to zero
            req_focus = np.where(req_focus_mask)[0]
            if tb_rule == "random":
                np.random.shuffle(req_focus)
                actions[req_focus[budget:]] = 0
            elif tb_rule == "ID":
                req_focus = np.sort(req_focus)
                actions[req_focus[budget:]] = 0
            elif tb_rule == "priority":
                # find priority of arms using tb_priority
                arm_priorities = tb_priority[cur_states]
                # focus on the arms outside focus set and requesting pulls
                arm_priorities = arm_priorities * req_focus_mask
                sorted_indices = np.argsort(arm_priorities)
                actions[sorted_indices[(self.N-(num_req_fs-budget)):]] = 0 ####
            else:
                raise NotImplementedError
            actions[non_focus] = 0
        elif num_non_req_fs >= (self.N - budget):
            # rectify focus set by setting some of the actions from one to zero; set other actions to one
            non_req_focus = np.where(non_req_focus_mask)[0]
            if tb_rule == "random":
                np.random.shuffle(non_req_focus)
                actions[non_req_focus[(self.N - budget):]] = 1
            elif tb_rule == "ID":
                non_req_focus = np.sort(non_req_focus)
                actions[non_req_focus[(self.N - budget):]] = 1
            elif tb_rule == "priority":
                # find priority of arms using tb_priority
                arm_priorities = tb_priority[cur_states]
                # focus on the arms outside focus set and requesting pulls
                arm_priorities = arm_priorities * non_req_focus_mask
                sorted_indices = np.argsort(arm_priorities)
                actions[sorted_indices[self.N-(num_non_req_fs+budget-self.N):]] = 1
                # print(num_non_req_fs - (self.N - budget), len(sorted_indices[-(num_non_req_fs+budget-self.N):]))
            else:
                raise NotImplementedError
            actions[non_focus] = 1
        else:
            # no rectify focus set; take suitable number of actions to zero and one
            # print("case 3: actions before rect=", actions)
            ## tie-breaking based on ID
            if num_req > budget:
                req_non_focus_mask = actions * (1-cur_focus_set_mask)
                req_non_focus = np.where(req_non_focus_mask)[0]
                if tb_rule == "random":
                    np.random.shuffle(req_non_focus)
                    actions[req_non_focus[(budget-num_req_fs):]] = 0
                elif tb_rule == "ID":
                    req_non_focus = np.sort(req_non_focus)
                    actions[req_non_focus[(budget-num_req_fs):]] = 0
                elif tb_rule == "priority":
                    # find priority of arms using tb_priority
                    arm_priorities = tb_priority[cur_states]
                    # focus on the arms outside focus set and requesting pulls
                    arm_priorities = arm_priorities * req_non_focus_mask
                    sorted_indices = np.argsort(arm_priorities)
                    actions[sorted_indices[-(num_req-budget):]] = 0
                else:
                    raise NotImplementedError
            elif budget > num_req:
                non_req_non_focus_mask = (1-actions) * (1-cur_focus_set_mask)
                non_req_non_focus = np.where(non_req_non_focus_mask)[0]
                if tb_rule == "random":
                    np.random.shuffle(non_req_non_focus)
                    actions[non_req_non_focus[(self.N-budget-num_non_req_fs):]] = 1
                elif tb_rule == "ID":
                    req_non_focus = np.sort(non_req_non_focus)
                    actions[non_req_non_focus[(self.N-budget-num_non_req_fs):]] = 1
                elif tb_rule == "priority":
                    # find priority of arms using tb_priority
                    arm_priorities = tb_priority[cur_states]
                    # focus on the arms outside focus set and requesting pulls
                    arm_priorities = arm_priorities * non_req_non_focus_mask
                    sorted_indices = np.argsort(arm_priorities)
                    actions[sorted_indices[-(budget-num_req):]] = 1 ####
                else:
                    raise NotImplementedError
            else:
                pass
            conformity_flag = 1
        assert np.sum(actions) == budget, "np.sum(actions)={}, budget={}".format(np.sum(actions), budget)

        return actions, conformity_flag

    def get_new_focus_set_two_stage(self, cur_states, subproblem="L1"):
        s2indices = {s: None for s in self.sspa}
        s_count_scaled = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(cur_states == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
        # print("Xt([N]) = {}".format(s_count_scaled))
        self.s_count_scaled.value = s_count_scaled

        if subproblem == "L1":
            constrs = []
            # the first round, solve under the L1 constraint
            constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= 0)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(self.z - self.m*self.state_probs <= self.d)
            constrs.append(- self.z + self.m*self.state_probs <= self.d)
            ## 0617 update: multiply 0.5 in from of the L1 norm
            constrs.append(0.5*cp.sum(self.d) <= self.beta * (1-self.m))
            objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
            problem = cp.Problem(objective, constrs)
            problem.solve()
            # the second round, solve under the c constraint, but contain the first solution
            z_inner = self.z.value.copy()
            constrs = []
            constrs.append(cp.SOC(self.f, self.W_sqrt @ (self.z - self.m * self.state_probs)))
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.z >= z_inner)  # the constraint that the new sol must contain the solution from last round
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(np.matmul(self.cpibs.T, self.z) <= self.act_frac)
            constrs.append(np.matmul(self.cpibs.T, self.z) >= self.act_frac - (1-self.m))
            objective = cp.Minimize(self.f + (self.Lw+0.1)*(1 - self.m))
            problem = cp.Problem(objective, constrs)
            problem.solve()
            z_outer = self.z.value.copy()
        else:
            raise NotImplementedError

        # two disjoint sets
        next_focus_set = []
        next_focus_set_outer = []
        for s in self.sspa:
            next_s_count_fs = int(self.N * z_inner[s])
            next_focus_set.extend(s2indices[s][0:next_s_count_fs])
            next_s_count_fs_outer = int(self.N * z_outer[s])
            next_focus_set_outer.extend(s2indices[s][next_s_count_fs:next_s_count_fs_outer])

        print(s_count_scaled, z_inner*(z_inner>self.EPS), z_outer*(z_outer>self.EPS))

        next_focus_set = np.array(next_focus_set, dtype=int)
        next_focus_set_outer = np.array(next_focus_set_outer, dtype=int)
        return next_focus_set, next_focus_set_outer

    def get_actions_two_stage(self, cur_states, cur_focus_set, cur_focus_set_outer):
        s2indices = {state:None for state in self.sspa}
        # count the indices of arms in each state
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]
        actions = np.zeros((self.N,), dtype=np.int64)
        for state in self.sspa:
            actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])

        budget = int(self.N * self.act_frac)
        budget += np.random.binomial(1, self.N * self.act_frac - budget)  # randomized rounding when alpha N

        # priority levels
        priority_levels = 3*np.ones((self.N,))
        priority_levels[cur_focus_set] = 1
        if len(cur_focus_set_outer) > 0:
            priority_levels[cur_focus_set_outer] = 2

        num_requests = np.sum(actions)
        conformity_flag = 0
        if num_requests > budget:
            # priority: action 0 > action 1 in focus set > action 1 in outer focus set > action 1 in none-focus set
            priority_level_w_requests = priority_levels * actions
            # we intentionally use random tie-breaking, to avoid ID from secretly playing a role
            random_tb_seed = 0.5*np.random.random(self.N)
            priority_level_w_requests += random_tb_seed
            sorted_indices = np.argsort(priority_level_w_requests)
            actions[sorted_indices[-int(num_requests - budget):]] = 0
            if np.sum(priority_level_w_requests >= 2) >= num_requests - budget:
                conformity_flag = 1
        elif num_requests < budget:
            # priority: action 1 > action 0 in focus set > action 0 in outer focus set > action 0 in none-focus set
            priority_level_no_requests = priority_levels * (1-actions)
            random_tb_seed = 0.5*np.random.random(self.N)
            priority_level_no_requests += random_tb_seed
            sorted_indices = np.argsort(priority_level_no_requests)
            actions[sorted_indices[-int(budget - num_requests):]] = 1
            if np.sum(priority_level_no_requests >= 2) >= budget - num_requests:
                conformity_flag = 1
        else:
            conformity_flag = 1
        assert np.sum(actions) == budget

        return actions, conformity_flag


class TwoSetPolicy(object):
    def __init__(self, sspa_size, y, N, act_frac, U, rounding="direct"):
        """
        :param rounding: "direct" or "misocp". If "direct", solve for a real-valued z when getting the focus set, and
         do the lower rounding to get the focus set; if "misocp", solve for a mixed integer qudratic program to get the focus set.
        """
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa = np.array([0, 1])
        self.sa_pairs = []
        for action in self.aspa:
            self.sa_pairs = self.sa_pairs + [(state, action) for state in self.sspa]

        self.N = N
        self.act_frac = act_frac
        self.y = y
        self.U = U
        self.U_sqrt = scipy.linalg.sqrtm(U)
        self.EPS = 1e-7
        self.rounding = rounding

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

        self.Sempty = np.where(self.state_probs <= self.EPS)[0]
        self.Sneu = np.where(np.all(self.y > self.EPS, axis=1))[0]
        self.eta = self.compute_eta()
        # todo: handle the case when y corresponds to more than one neutral state
        assert len(self.Sneu) <= 1

        # variables and parameters
        if rounding == "direct":
            self.z = cp.Variable(self.sspa_size)
        elif rounding == "misocp":
            self.Nz = cp.Variable(self.sspa_size, integer=True)
        else:
            raise NotImplementedError
        self.m = cp.Variable()
        self.f = cp.Variable()
        self.s_count_scaled = cp.Parameter(self.sspa_size)
        self.s_count_scaled_fs = cp.Parameter(self.sspa_size)
        self.beta = cp.Parameter()
        self.beta.value = min(act_frac, 1-act_frac)

    def compute_eta(self):
        if (self.U is np.infty) or len(self.Sneu) == 0:
            return 0

        c_act_no_neu = self.policy[:,1].copy()
        c_act_no_neu[self.Sneu[0]] = 0
        c_pass_no_neu = self.policy[:,0].copy()
        c_pass_no_neu[self.Sneu[0]] = 0

        x = cp.Variable(self.sspa_size)
        f_local = cp.Variable()
        constrs = []
        constrs.append(cp.sum(x) == 1)
        constrs.append(x >= 0)
        constrs.append(cp.SOC(f_local, self.U_sqrt @ (x-self.state_probs)))
        constrs.append(cp.matmul(c_act_no_neu.T, x) >= self.act_frac)
        objective = cp.Minimize(f_local)
        problem = cp.Problem(objective, constrs)
        problem.solve()
        if f_local.value is not None:
            f_1 = f_local.value.copy()
        else:
            f_1 = np.infty
        # solve the second problem
        constrs[-1] = cp.matmul(c_pass_no_neu.T, x) >= 1 - self.act_frac
        problem = cp.Problem(objective, constrs)
        problem.solve()
        if f_local.value is not None:
            f_2 = f_local.value.copy()
        else:
            f_2 = np.infty
        # todo: before running misocp, integrate the minus something back to the optimization problem
        return min(f_1, f_2) - (len(self.Sempty)+1)/self.N

    def get_new_focus_set(self, cur_states, last_OL_set):
        if (self.U is np.infty) or self.eta <= 0:
            return np.array([])

        states_fs = cur_states[last_OL_set]
        s2indices = {s: None for s in self.sspa}
        s2indices_fs = {s: None for s in self.sspa} # state to indices map in the focus set
        s_count_scaled = np.zeros((self.sspa_size,))
        s_count_scaled_fs = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s2indices[s] = np.where(cur_states == s)[0]
            s2indices_fs[s] = np.where(states_fs == s)[0]
            s_count_scaled[s] = len(s2indices[s]) / self.N
            s_count_scaled_fs[s] = len(s2indices_fs[s]) / self.N
        # print("Xt([N]) = {}, Xt(D(t-1)) = {}".format(s_count_scaled, s_count_scaled_fs))
        self.s_count_scaled.value = s_count_scaled
        self.s_count_scaled_fs.value = s_count_scaled_fs

        cur_m = len(last_OL_set)/self.N
        # first solve for the D^{OL}_temp
        x_minus_mmu = s_count_scaled_fs - cur_m * self.state_probs
        cur_delta = self.eta*cur_m - np.sqrt(np.matmul(np.matmul(x_minus_mmu.T, self.U), x_minus_mmu))
        if cur_delta >= 0:
            z_temp = s_count_scaled_fs
        else:
            # shrink the OL set
            if self.rounding == "direct":
                constrs = []
                constrs.append(self.z >= 0)
                constrs.append(self.z <= self.s_count_scaled_fs)
                constrs.append(self.m == cp.sum(self.z))
                constrs.append(cp.SOC(self.eta*self.m, self.U_sqrt @ (self.z - self.m*self.state_probs)))
                objective = cp.Maximize(self.m)
                problem = cp.Problem(objective, constrs)
                problem.solve()
                z_temp = self.z.value.copy()
            elif self.rounding == "misocp":
                constrs = []
                constrs.append(self.Nz >= 0)
                constrs.append(self.Nz <= self.N*self.s_count_scaled_fs)
                constrs.append(self.m == cp.sum(self.Nz) / self.N)
                constrs.append(cp.SOC(self.eta*self.m, self.U_sqrt @ (self.Nz / self.N - self.m*self.state_probs)))
                objective = cp.Maximize(self.m)
                problem = cp.Problem(objective, constrs)
                problem.solve()
                z_temp = self.Nz.value.copy() / self.N
            else:
                raise NotImplementedError
        # then solve for the next OL set
        if self.rounding == "direct":
            constrs = []
            constrs.append(self.z >= z_temp)
            constrs.append(self.z <= self.s_count_scaled)
            constrs.append(self.m == cp.sum(self.z))
            constrs.append(cp.SOC(self.eta*self.m, self.U_sqrt @ (self.z - self.m*self.state_probs)))
            objective = cp.Maximize(self.m)
            problem = cp.Problem(objective, constrs)
            problem.solve()
            z_OL = self.z.value.copy()
        elif self.rounding == "misocp":
            constrs = []
            constrs.append(self.Nz >= self.N*z_temp)
            constrs.append(self.Nz <= self.N*self.s_count_scaled)
            constrs.append(self.m == cp.sum(self.Nz) / self.N)
            constrs.append(cp.SOC(self.eta*self.m, self.U_sqrt @ (self.Nz / self.N - self.m*self.state_probs)))
            objective = cp.Maximize(self.m)
            problem = cp.Problem(objective, constrs)
            problem.solve()
            z_OL = self.z.value.copy()
        else:
            raise NotImplementedError

        next_OL_set = []
        for s in self.sspa:
            next_s_count_fs = int(self.N * z_OL[s])
            next_OL_set.extend(s2indices[s][0:next_s_count_fs])
        next_OL_set = np.array(next_OL_set, dtype=int)
        # print("state count = ", self.z.value * self.N, "states = ", cur_states, "focus set = ", next_focus_set)
        return next_OL_set

    def get_actions(self, cur_states, cur_OL_set):
        exp_budget = self.N * self.act_frac
        random_bit = np.random.binomial(1, exp_budget - int(exp_budget))
        budget = int(exp_budget) + random_bit

        s2indices = {state:None for state in self.sspa}
        for state in self.sspa:
            s2indices[state] = np.where(cur_states == state)[0]

        actions = np.zeros((self.N,), dtype=np.int64)
        if len(cur_OL_set) > 0:
            local_exp_budget = len(cur_OL_set) * self.act_frac
            # todo: different from the paper, here we allow non-integer alpha N;
            #  and we share the random bit between budget and local budget so that action recfitication always work.
            #  Check if the proof still goes through in this setting, or just avoid non-integer alpha N.
            local_budget = int(local_exp_budget) + random_bit

            s2indices_fs = {state:None for state in self.sspa}
            cur_OL_set_mask = np.zeros((self.N,))
            cur_OL_set_mask[cur_OL_set] = 1
            # count the indices of arms in each state
            for state in self.sspa:
                s2indices_fs[state] =  np.where(np.all([cur_states == state, cur_OL_set_mask], axis=0))[0]
                expected_act_count = self.policy[state, 1] * len(s2indices_fs[state])
                # choose actions by randomized rounding
                if state in self.Sempty:
                    p_round = expected_act_count - int(expected_act_count)
                    act_count = int(expected_act_count) + np.random.binomial(1, p_round)
                elif state in self.Sneu:
                    continue
                else:
                    act_count = int(expected_act_count)
                actions[s2indices_fs[state][0:act_count]] = 1
            rem_local_budget = local_budget - np.sum(actions)
            num_neutral_arms = len(s2indices_fs[self.Sneu[0]])
            # if self.rounding == "direct":
            if rem_local_budget < 0:
                # print("baka", end=" ")
                # activate no neutral arms; rectify some arms that take active actions;
                actions[s2indices_fs[self.Sneu[0]]] = 0
                act_arms = np.where(np.all([actions == 1, cur_OL_set_mask], axis=0))[0]
                actions[act_arms[0:(-rem_local_budget)]] = 0
            elif rem_local_budget > num_neutral_arms:
                # print("baka", end=" ")
                # activate all neutral arms; recfity some arms that take passive actions;
                actions[s2indices_fs[self.Sneu[0]]] = 1
                pas_arms = np.where(np.all([actions == 0, cur_OL_set_mask], axis=0))[0]
                actions[pas_arms[0:(rem_local_budget - num_neutral_arms)]] = 1
            else:
                actions[s2indices_fs[self.Sneu[0]][0:rem_local_budget]] = 1
            # elif self.rounding == "misocp":
            #     assert rem_local_budget >= 0
            #     assert rem_local_budget <= num_neutral_arms, "{}>{}".format(rem_local_budget, num_neutral_arms)
            #     actions[s2indices_fs[self.Sneu[0]][0:rem_local_budget]] = 1
            # else:
            #     raise NotImplementedError
            assert np.sum(actions[cur_OL_set]) == local_budget, "Error: {}!={}".format(np.sum(actions[cur_OL_set]), local_budget)

        # for this version, just use ID policy for the actions outside the OL set
        non_OL_set_mask = np.ones((self.N,))
        if len(cur_OL_set) > 0:
            non_OL_set_mask[cur_OL_set] = 0
        non_OL_set = np.where(non_OL_set_mask)[0]
        ideal_actions = np.zeros((self.N,))
        for state in self.sspa:
            ideal_actions[s2indices[state]] = np.random.choice(self.aspa, size=len(s2indices[state]), p=self.policy[state])
        actions[non_OL_set] = ideal_actions[non_OL_set]
        # rectification
        num_requests = np.sum(actions)
        if num_requests > budget:
            indices_request = np.where(actions*non_OL_set_mask)[0]
            # sort by ID
            indices_request = np.sort(indices_request)
            request_ignored = indices_request[(-int(num_requests - budget)):]
            actions[request_ignored] = 0
        elif num_requests < budget:
            indices_no_request = np.where((1-actions)*non_OL_set_mask)[0]
            # sort by ID
            indices_no_request = np.sort(indices_no_request)
            no_request_pulled = indices_no_request[(-int(budget - num_requests)):]
            actions[no_request_pulled] = 1
        else:
            pass
        assert np.sum(actions) == budget, "{}!={}".format(np.sum(actions), budget)
        return actions

def states_to_scaled_state_counts(sspa_size, N, states):
    scaled_state_counts = np.zeros((sspa_size,)) # 2 is the action space size
    for i in range(len(states)):
        s = int(states[i])
        scaled_state_counts[s] += 1
    return scaled_state_counts / N

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
