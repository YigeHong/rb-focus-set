import numpy as np
import pickle
from discrete_RB import SingleArmAnalyzer
from matplotlib import pyplot as plt
import warnings

class Gast20Example1(object):
    def __init__(self):
        self.sspa_size = 3
        # self.sspa = np.array(list(range(self.sspa_size)))
        # self.aspa_size = 2
        # self.aspa = np.array(list(range(self.aspa_size)))
        P0 = [[0.5214073, 0.40392496, 0.07466774],
              [0.0158415, 0.21455666, 0.76960184],
              [0.53722329, 0.37651148, 0.08626522]]
        P1 = [[0.24639364, 0.23402385, 0.51958251],
              [0.49681581, 0.49509821, 0.00808597],
              [0.37826553, 0.15469252, 0.46704195]]
        R0 = [0, 0, 0]
        R1 = [0.72232506, 0.18844869, 0.25752477]
        self.trans_tensor = np.array([P0, P1]).transpose((1,0,2)) # dimensions (state, action, next_state)
        self.reward_tensor = np.array([R0, R1]).transpose((1,0)) # dimensions (state, action)

        self.suggest_act_frac = 0.4


class Gast20Example2(object):
    def __init__(self):
        self.sspa_size = 3
        # self.sspa = np.array(list(range(self.sspa_size)))
        # self.aspa_size = 2
        # self.aspa = np.array(list(range(self.aspa_size)))
        P0 = [[0.02232142, 0.10229283, 0.87538575],
              [0.03426605, 0.17175704, 0.79397691],
              [0.52324756, 0.45523298, 0.02151947]]
        P1 = [[0.14874601, 0.30435809, 0.54689589],
              [0.56845754, 0.41117331, 0.02036915],
              [0.25265570, 0.27310439, 0.4742399]]
        R0 = [0, 0, 0]
        R1 = [0.37401552, 0.11740814, 0.07866135]
        self.trans_tensor = np.array([P0, P1]).transpose((1,0,2)) # dimensions (state, action, next_state)
        self.reward_tensor = np.array([R0, R1]).transpose((1,0)) # dimensions (state, action)

        self.suggest_act_frac = 0.4


class Gast20Example3(object):
    def __init__(self):
        self.sspa_size = 3
        # self.sspa = np.array(list(range(self.sspa_size)))
        # self.aspa_size = 2
        # self.aspa = np.array(list(range(self.aspa_size)))
        P0 = [[0.47819592, 0.02090623, 0.50089785],
              [0.08063373, 0.15456935, 0.76479692],
              [0.66552514, 0.08481946, 0.2496554]]
        P1 = [[0.00279465, 0.37327924, 0.62392611],
              [0.51582335, 0.46333908, 0.02083756],
              [0.41875202, 0.17776712, 0.40348086]]
        R0 = [0, 0, 0]
        R1 = [0.97658608, 0.53014109, 0.40394919]
        self.trans_tensor = np.array([P0, P1]).transpose((1,0,2)) # dimensions (state, action, next_state)
        self.reward_tensor = np.array([R0, R1]).transpose((1,0)) # dimensions (state, action)

        self.suggest_act_frac = 0.4


class RandomExample(object):
    def __init__(self, sspa_size, distr, verbose=False, laziness=None, parameters=None):
        """
        :param sspa_size:
        :param distr: string, "uniform", or  "beta05" or "beta0"
        """
        self.sspa_size = sspa_size
        # self.sspa = np.array(list(range(self.sspa_size)))
        # self.aspa_size = 2
        # self.aspa = np.array(list(range(self.aspa_size)))
        self.distr = distr
        if distr == "uniform":
            P0 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
            P1 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
            R0 = np.zeros((self.sspa_size,))
            R1 = np.random.uniform(0, 1, size=(self.sspa_size, ))
        elif distr == "uniform-nzR0":
            P0 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
            P1 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
            R0 = np.random.uniform(0, 1, size=(self.sspa_size, ))
            R1 = np.random.uniform(0, 1, size=(self.sspa_size, ))
        elif distr == "uniform-simplex":
            P0 = np.random.dirichlet([1]*sspa_size, self.sspa_size)
            P1 = np.random.dirichlet([1]*sspa_size, self.sspa_size)
            R0 = np.zeros((self.sspa_size,))
            R1 = np.random.uniform(0, 1, size=(self.sspa_size, ))
        # elif distr == "uniform-symR":
        #     P0 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
        #     P1 = np.random.uniform(0, 1, size=(self.sspa_size, self.sspa_size))
        #     R0 = np.random.uniform(-1, 1, size=(self.sspa_size, ))
        #     R1 = np.random.uniform(-1, 1, size=(self.sspa_size, ))
        elif distr == "beta05":
            P0 = np.random.beta(0.5, 0.5, size=(self.sspa_size, self.sspa_size))
            P1 = np.random.beta(0.5, 0.5, size=(self.sspa_size, self.sspa_size))
            R0 = np.zeros((self.sspa_size,))
            R1 = np.random.beta(0.5, 0.5, size=(self.sspa_size, ))
        elif distr == "beta0":
            P0 = np.random.beta(0, 0, size=(self.sspa_size, self.sspa_size))
            P1 = np.random.beta(0, 0, size=(self.sspa_size, self.sspa_size))
            R0 = np.zeros((self.sspa_size,))
            R1 = np.random.beta(0, 0, size=(self.sspa_size, ))
        elif distr == "dirichlet":
            if len(parameters) == 1:
                alphas = [parameters[0]] * self.sspa_size
            else:
                alphas = parameters
            P0 = np.random.dirichlet(alphas, self.sspa_size)
            P1 = np.random.dirichlet(alphas, self.sspa_size)
            R0 = np.random.dirichlet(alphas, 1)[0,:]
            R1 = np.random.dirichlet(alphas, 1)[0,:]
            P0 = P0 * (P0 >= 1e-7)
            P1 = P1 * (P1 >= 1e-7)
        else:
            raise NotImplementedError
        P0 = P0 / np.sum(P0, axis=1, keepdims=True)
        P1 = P1 / np.sum(P1, axis=1, keepdims=True)
        if laziness is not None:
            P0 = (1-laziness) * np.eye(self.sspa_size) + laziness * P0
            P1 = (1-laziness) * np.eye(self.sspa_size) + laziness * P1
        self.trans_tensor = np.stack([P0, P1], axis=1) # dimensions (state, action, next_state)
        self.reward_tensor = np.stack([R0, R1], axis=1) # dimensions (state, action)
        # make sure alpha is not too close to 0 or 1; round to 0.01
        self.suggest_act_frac = int(100*np.random.uniform(0.1,0.9))/100

        self.unichain_eigval = None
        self.local_stab_eigval = None
        self.y = None
        self.avg_reward_upper_bound = None
        self.lp_priority = None
        self.whittle_priority = None
        self.avg_reward_lpp_mf_limit = None
        self.avg_reward_whittle_mf_limit = None
        self.reward_modifs = []
        self.sa_max_hitting_time = None

        if verbose:
            print("P0 = ", P0)
            print("P1 = ", P1)
            print("R0 = ", R0)
            print("R1 = ", R1)

class RewardModification(object):
    def __init__(self):
        self.reward_tensor = None
        self.avg_reward_mf_limit = None
        self.avg_reward_upper_bound = None


class GeometricRandomExample(object):
    def __init__(self, sspa_size, point_distr, edge_distr, threshold, reward_distr, verbose=False, laziness=None, parameters=None):
        self.sspa_size = sspa_size
        self.point_distr = point_distr
        self.edge_distr = edge_distr
        self.threshold = threshold
        self.reward_distr = reward_distr
        self.trans_tensor = np.zeros((sspa_size, 2, sspa_size))
        self.parameters = parameters
        if point_distr == "square":
            if edge_distr == "uniform":
                # for each action, generate a random graph
                coordinates = []
                # assume connected to oneself
                adj_table = np.eye(sspa_size)
                for i in range(sspa_size):
                    coordinates.append(np.random.uniform(0,1,[2]))
                coordinates = np.array(coordinates)
                self.coordinates = coordinates
                for i in range(sspa_size):
                    for j in range(sspa_size):
                        dist_ij = np.linalg.norm(coordinates[i,:] - coordinates[j,:])
                        if dist_ij < threshold:
                            adj_table[i,j] = 1
                for i in range(sspa_size):
                    num_neighbors = int(np.sum(adj_table[i,:]))
                    neighbor_inds = np.where(adj_table[i,:])[0]
                    for a in range(2):
                        # uniform distribution on the simplex
                        probs_on_neighbors = np.random.dirichlet([1]*num_neighbors, 1)
                        # setting the probabilities
                        self.trans_tensor[i,a, neighbor_inds] = probs_on_neighbors
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if reward_distr == "uniform":
            self.reward_tensor = np.random.uniform(0, 1, (sspa_size, 2))
        elif reward_distr == "dirichlet":
            alphas = [parameters["reward_alpha"]] * sspa_size
            self.reward_tensor = np.random.dirichlet(alphas, 2).T
        else:
            raise NotImplementedError
        self.suggest_act_frac = int(100*np.random.uniform(0.1,0.9))/100

    def save(self, f_path, other_params=None):
        data_dict = {"sspa_size": self.sspa_size,
         "trans_tensor": self.trans_tensor,
         "reward_tensor": self.reward_tensor,
         "point_distr": self.point_distr,
         "edge_distr": self.edge_distr,
         "threshold": self.threshold,
         "reward_distr": self.reward_distr,
         # "all_action_coordinates": setting.all_action_coordinates,
         "coordinates": self.coordinates,
         "suggest_act_frac": self.suggest_act_frac,
          "parameters": self.parameters
         }
        if other_params is not None:
            data_dict.update(other_params)
        with open(f_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def visualize_graph(self):
        # fig, axes = plt.subplots(1, 2)
        # for a in range(2):
        #     for i in range(self.sspa_size):
        #         for j in range(self.sspa_size):
        #             coordinates = self.all_action_coordinates[0]
        #             if self.trans_tensor[i,a,j] > 0:
        #                 axes[a].plot([coordinates[i,0], coordinates[j,0]], [coordinates[i,1], coordinates[j,1]])
        # plt.show()
        fig, axes = plt.subplots(1, 2)
        for a in range(2):
            for i in range(self.sspa_size):
                for j in range(self.sspa_size):
                    coordinates = self.coordinates
                    if self.trans_tensor[i,a,j] > 0.1:
                        axes[a].plot([coordinates[i,0], coordinates[j,0]], [coordinates[i,1], coordinates[j,1]])
        plt.show()


def ChungLuBeta2B(beta):
    zeta = 0
    num_terms_zeta = int((1e-8)**(-1/(beta-1)))
    for k in range(1, num_terms_zeta):
        zeta += k**(-beta)
    B = 1 / ((beta-1)*zeta)
    return B

class ChungLuRandomExample(object):
    def __init__(self, sspa_size, beta, B):
        self.sspa_size = sspa_size
        self.beta = beta
        self.B = B
        self.weights = np.array([((i+1)/(sspa_size*B))**(-1/(beta-1)) for i in range(self.sspa_size)])
        print("CL model vertex weights=", self.weights)
        self.total_weights = np.sum(self.weights)
        self.trans_tensor = np.zeros((sspa_size, 2, sspa_size))

        self.adj_tables = []
        for a in range(2):
            # assume connected to oneself
            adj_table = np.eye(sspa_size)
            for i in range(sspa_size):
                for j in range(sspa_size):
                    adj_table[i,j] = np.random.binomial(1, self.weights[i]*self.weights[j] / self.total_weights)
            self.adj_tables.append(adj_table)
            for i in range(sspa_size):
                num_neighbors = int(np.sum(adj_table[i,:]))
                neighbor_inds = np.where(adj_table[i,:])[0]
                # uniform distribution on the simplex
                probs_on_neighbors = np.random.dirichlet([1]*num_neighbors, 1)
                # setting the probabilities
                self.trans_tensor[i, a, neighbor_inds] = probs_on_neighbors

        self.reward_tensor = np.random.dirichlet([1]*sspa_size, 2).T
        self.suggest_act_frac = int(100*np.random.uniform(0.1,0.9))/100

        self.unichain_eigval = None
        self.local_stab_eigval = None
        self.avg_reward_upper_bound = None
        self.lp_priority = None
        self.whittle_priority = None
        self.avg_reward_lpp_mf_limit = None
        self.avg_reward_whittle_mf_limit = None
        self.reward_modifs = []
        sa_max_hitting_time = None


class ExampleFromFile(object):
    def __init__(self, f_path):
        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.sspa_size = data_dict["sspa_size"]
        self.trans_tensor = data_dict["trans_tensor"]
        self.reward_tensor = data_dict["reward_tensor"]
        # self.distr = data_dict["distr"]
        self.suggest_act_frac = data_dict["suggest_act_frac"]


class ConveyorExample(object):
    """
    Imagine a conveyor that moves arms from left to right
    Each position specifies an action 0 or 1.
    The arm has to take the right action to move to right;
    if the action is wrong, move back 
    When an arm reaches the right most point, receive a reward and start over from the left most point
    We left half all require 0, and right half all require 1
    0 0 0 0 0 1 1 1 1 1
    """
    def __init__(self, sspa_size, probs_L, probs_R, action_script, suggest_act_frac, r_disturb=None):
        """
        :param sspa_size: total number of states of the system. The left half requires 0, the right half requires 1
        :param probs_L: prob of moving left if take the wrong action
        :param probs_R: prob of moving right if take the correct action
        """
        assert sspa_size % 2 == 0, "state space size should be an even number"
        assert (len(probs_L) == sspa_size) and (len(probs_R) == sspa_size), "invalid size of probs_L and probs_R lists"
        self.sspa_size = sspa_size
        self.probs_L = probs_L.copy()
        self.probs_R = probs_R.copy()

        self.action_script = action_script.copy()
        self.suggest_act_frac = suggest_act_frac

        # define the transition kernel
        self.trans_tensor = np.zeros((sspa_size, 2, sspa_size))
        for s in range(self.sspa_size):
            action_R = self.action_script[s]
            action_L = 1 - action_R
            if s == 0:
                self.trans_tensor[s, action_R, s+1] = self.probs_R[s]
                self.trans_tensor[s, action_R, s] = 1 - self.probs_R[s]
                # reflect at the left boundary, so we use self.probs_L[s] to specify the prob of going RIGHT if take the wrong action
                self.trans_tensor[s, action_L, s+1] = self.probs_L[s]
                self.trans_tensor[s, action_L, s] = 1 - self.probs_L[s]
            elif s == self.sspa_size - 1:
                # restart at 0 if go beyond the right boundary
                self.trans_tensor[s, action_R, 0] = self.probs_R[s]
                self.trans_tensor[s, action_R, s] = 1 - self.probs_R[s]
                self.trans_tensor[s, action_L, s-1] = self.probs_L[s]
                self.trans_tensor[s, action_L, s] = 1 - self.probs_L[s]
            else:
                self.trans_tensor[s, action_R, s+1] = self.probs_R[s]
                self.trans_tensor[s, action_R, s] = 1 - self.probs_R[s]
                self.trans_tensor[s, action_L, s-1] = self.probs_L[s]
                self.trans_tensor[s, action_L, s] = 1 - self.probs_L[s]

        # define the reward. It is zero in most of the situations
        self.reward_tensor = np.zeros((sspa_size, 2))
        # if take the correct action in the last state, receive reward 1 with the probability of moving to the right
        last_index = self.sspa_size - 1
        self.reward_tensor[last_index, self.action_script[last_index]] = 1 * self.probs_R[last_index]
        if r_disturb is not None:
            self.reward_tensor[0,1] = r_disturb

    @staticmethod
    def get_parameters(name, sspa_size, nonindexable=False):
        """
        :param name:
        :param sspa_size: size of the mdp
        :return: probs_L, probs_R
        """
        assert sspa_size % 2 == 0, "state-space size should be an even number"
        half_size = int(sspa_size / 2)
        if name == "eg4action-gap-tb":
            """
            The purpose of this counterexample is to let the LP-Priority perform poorly 
            To do this, 
            action script 1 1 1 1 0 0 0 0 (or it can be longer)
            keep probs_R[i] constant for each i for convenience 
            probs_L should be strictly decreasing on the places where the action script is 1, so that the priority is increasing
            """
            probs_R = np.ones((sspa_size, )) * 0.1
            probs_L = np.ones((sspa_size, )) * (0.5 - 0.01*np.arange(0, sspa_size))
            probs_L[1] = 1.0 #1/6 - 0.01
            probs_L[0] = 0.0 # 1/3 - 0.01
            action_script = np.zeros((sspa_size, ), dtype=np.int64)
            action_script[:half_size] = 1
            tries_in_a_loop = 1 / probs_R
            suggest_act_frac = np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
            if nonindexable:
                r_disturb = 0.04 / sspa_size
                assert r_disturb < 11*suggest_act_frac / (50*sspa_size)
            else:
                r_disturb = 0
        elif name == "arbitrary-length":
            probs_R = np.ones((sspa_size, )) * 0.1
            probs_L = np.ones((sspa_size, )) * (0.5 - 0.1*np.arange(0, sspa_size)/sspa_size)
            probs_L[1] = 1.0
            probs_L[0] = 0.0
            action_script = np.zeros((sspa_size, ), dtype=np.int64)
            action_script[:half_size] = 1
            # we choose it to be slightly less than 0.5
            suggest_act_frac = 0.4
            if nonindexable:
                r_disturb = 0.04 / sspa_size
                assert r_disturb < 11*suggest_act_frac / (50*sspa_size)
            else:
                r_disturb = 0
        else:
            warnings.warn("The RB setting {} is depricated.".format(name))
            if name == "eg4archive1":
                """
                The following set of parameters are producing ridiculous outcomes: 
                LP performance suddenly jumps from 0 to optimal when N=1000
                To reproduce, use random seed 0; 
                initialize the arm evenly using the code
                for i in range(N):
                    init_states[i] = i % sspa_size
                """
                probs_R = np.ones((sspa_size, )) / 6
                probs_L = np.ones((sspa_size, )) * 0.9
                probs_L[0] = 1.0
                # produce the action script
                action_script = np.zeros((sspa_size, ), dtype=np.int64)
                action_script[:half_size] = 1
                tries_in_a_loop = 1 / probs_R
                suggest_act_frac = np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
            elif name == "eg4archive2":
                """
                this is a setting that makes LP bad if initialized properly
                sspa = 8
                do not fix dual variable
                """
                probs_R = np.ones((sspa_size, )) / 6
                probs_L = np.ones((sspa_size, )) / (2+0.1*sspa_size) # this is a bug, sspa_size should be arange(0, sspa_size)
                probs_L[1] = 1/2 #1/6 - 0.01
                probs_L[0] = 1/3 #1/3 - 0.01
                action_script = np.zeros((sspa_size, ), dtype=np.int64)
                action_script[:half_size] = 1
                tries_in_a_loop = 1 / probs_R
                suggest_act_frac = np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
            elif name == "eg4unif-tb":
                """
                The purpose of this counterexample is to let DRP with uniform tie breaking perform poorly
                To do this, 
                action script 1 1 1 1 0 0 0 (or it can be longer)
                keep probs_R[i] constant for each i for convenience
                probs_L should be strictly greater than probs_R on the place where the action script is 1, so that there is a drift to left
                """
                # each point has a 1/6 prob of moving to the right if the correct action is chosen
                probs_R = np.ones((sspa_size, )) / 6
                # define the prob of moving to the left.
                probs_L = np.ones((sspa_size, )) * 0.9
                probs_L[0] = 0.0   # reflection prob is zero
                # produce the action script, which defines the "right" action at each state
                action_script = np.zeros((sspa_size, ), dtype=np.int64)
                action_script[:half_size] = 1
                # produce the suggested activation fraction. It should be just enough to let the arm always move to the right
                tries_in_a_loop = 1 / probs_R
                suggest_act_frac = np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
            else:
                raise NotImplementedError
            r_disturb = 0
        return probs_L, probs_R, action_script, suggest_act_frac, r_disturb


class NonSAExample(object):
    def __init__(self):
        self.sspa_size = 8
        action_script = [1,1,1,1,1,1,0,1]
        self.trans_tensor = np.zeros((8, 2, 8))
        for i in range(6):
            self.trans_tensor[i, action_script[i], i+1] = 1
            self.trans_tensor[i, 1-action_script[i], 0] = 1

        self.trans_tensor[6, action_script[6], 7] = 1/2
        self.trans_tensor[6, action_script[6], 4] = 1/2
        self.trans_tensor[6, 1-action_script[6], 0] = 1

        self.trans_tensor[7, action_script[7], 6] = 1
        self.trans_tensor[7, 1-action_script[7], 0] = 1

        self.reward_tensor = np.zeros((8, 2))
        for i in [4,5,6,7]:
            self.reward_tensor[i, action_script[i]] = 1

        self.suggest_act_frac = 0.6


class BigNonSAExample(object):
    def __init__(self, version="v1"):
        if version == "v1":
            # no longer use this version
            raise NotImplementedError
            self.sspa_size = 11
            action_script = [1,0,1,0,1, 1,1,1,0,0,0]
            self.trans_tensor = np.zeros((11, 2, 11))
            for i in range(11):
                if i not in [8,10]:
                    self.trans_tensor[i, action_script[i], i+1] = 1
                    self.trans_tensor[i, 1-action_script[i], 0] = 1
                else:
                    continue

            self.trans_tensor[8, action_script[8], 9] = 1/2
            self.trans_tensor[8, action_script[8], 5] = 1/2
            self.trans_tensor[8, 1-action_script[8], 0] = 1

            self.trans_tensor[10, action_script[10], 8] = 1
            self.trans_tensor[10, 1-action_script[10], 0] = 1

            self.reward_tensor = np.zeros((11, 2))
            for i in [5,6,7,8,9,10]:
                self.reward_tensor[i, action_script[i]] = 1

            self.suggest_act_frac = 3/7
        if version == "v2":
            self.sspa_size = 12
            action_script = [1,0,1,0,1, 1,1,1,1,0,0,0]
            self.trans_tensor = np.zeros((12, 2, 12))
            for i in range(12):
                if i not in [9,11]:
                    self.trans_tensor[i, action_script[i], i+1] = 1
                    self.trans_tensor[i, 1-action_script[i], 0] = 1
                else:
                    continue

            self.trans_tensor[9, action_script[9], 10] = 1/2
            self.trans_tensor[9, action_script[9], 5] = 1/2
            self.trans_tensor[9, 1-action_script[9], 0] = 1

            self.trans_tensor[11, action_script[11], 9] = 1
            self.trans_tensor[11, 1-action_script[11], 0] = 1

            self.reward_tensor = np.zeros((12, 2))
            for i in [5,6,7,8,9,10,11]:
                self.reward_tensor[i, action_script[i]] = 1

            self.suggest_act_frac = 1/2
        elif version == "v3":
            self.sspa_size = 10
            action_script = [1,1,1,1,1,1,1,1,0,1]
            self.trans_tensor = np.zeros((10, 2, 10))
            for i in range(8):
                self.trans_tensor[i, action_script[i], i+1] = 1
                self.trans_tensor[i, 1-action_script[i], 0] = 1

            self.trans_tensor[8, action_script[8], 9] = 1/2
            self.trans_tensor[8, action_script[8], 6] = 1/2
            self.trans_tensor[8, 1-action_script[8], 0] = 1

            self.trans_tensor[9, action_script[9], 8] = 1
            self.trans_tensor[9, 1-action_script[9], 0] = 1

            self.reward_tensor = np.zeros((10, 2))
            for i in [6,7,8,9]:
                self.reward_tensor[i, action_script[i]] = 1

            self.suggest_act_frac = 0.6
        elif version == "v4":
            self.sspa_size = 10
            action_script = [1,1,1,0,0,0,1,1,0,1]
            self.trans_tensor = np.zeros((10, 2, 10))
            for i in range(8):
                self.trans_tensor[i, action_script[i], i+1] = 1
                self.trans_tensor[i, 1-action_script[i], 0] = 1

            self.trans_tensor[8, action_script[8], 9] = 1/2
            self.trans_tensor[8, action_script[8], 6] = 1/2
            self.trans_tensor[8, 1-action_script[8], 0] = 1

            self.trans_tensor[9, action_script[9], 8] = 1
            self.trans_tensor[9, 1-action_script[9], 0] = 1

            self.reward_tensor = np.zeros((10, 2))
            for i in [6,7,8,9]:
                self.reward_tensor[i, action_script[i]] = 1

            self.suggest_act_frac = 0.6
        else:
            raise NotImplementedError


class NonIndexableExample(object):
    def __init__(self):
        self.sspa_size = 3
        P0 = [[0, 0.4156, 0.3942],
              [0.5676, 0, 0.0133],
              [0.0191, 0.1097, 0]]
        P1 = [[0, 0.0903, 0.1301],
              [0.1903, 0, 0.6234],
              [0.2901, 0.3901, 0]]
        for i in range(3):
            P0[i][i] = 1 - np.sum(P0[i])
            P1[i][i] = 1 - np.sum(P1[i])
        R0 = [0.458, 0.5308, 0.6873]
        R1 = [0.9631, 0.7963, 0.1057]

        self.trans_tensor = np.array([P0, P1]).transpose((1,0,2)) # dimensions (state, action, next_state)
        self.reward_tensor = np.array([R0, R1]).transpose((1,0)) # dimensions (state, action)

        self.suggest_act_frac = 0.5 # it can be anything.


def print_bandit(setting, latex_format=False):
    #print("generated using {} distribution".format(self.distr))
    print("------------Information of the bandits---------------")
    print("state space size = ", setting.sspa_size)
    if not latex_format:
        print("P0 = \n", setting.trans_tensor[:,0,:])
        print("P1 = \n", setting.trans_tensor[:,1,:])
        print("R0 = \n", setting.reward_tensor[:,0])
        print("R1 = \n", setting.reward_tensor[:,1])
    else:
        tensors = {"P0":setting.trans_tensor[:,0,:], "P1":setting.trans_tensor[:,1,:],
                   "R0":setting.reward_tensor[:,0], "R1":setting.reward_tensor[:,1]}
        for name, data in tensors.items():
            print(name + " = ")
            if len(data.shape) == 1:
                for i in range(data.shape[0]):
                    print("{:.5}".format(data[i]), end=" ")
                    if i < data.shape[0]-1:
                        print("&", end=" ")
            elif len(data.shape) == 2:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        print("{:.5}".format(data[i,j]), end=" ")
                        if j < data.shape[0]-1:
                            print("&", end=" ")
                    if i < data.shape[1] -1:
                        print(r"\\", end="\n")
            else:
                raise NotImplementedError
            print()

    if hasattr(setting, "suggest_act_frac"):
        print("suggest act frac = ", setting.suggest_act_frac)
    print("---------------------------")


def save_bandit(setting, f_path, other_params):
    data_dict = {"sspa_size": setting.sspa_size,
         "trans_tensor": setting.trans_tensor,
         "reward_tensor": setting.reward_tensor,
         "distr": setting.distr,
         "suggest_act_frac": setting.suggest_act_frac
         }
    if other_params is not None:
        data_dict.update(other_params)
    with open(f_path, 'wb') as f:
        pickle.dump(data_dict, f)

def generate_Geo_random():
    # sspa_size = 10
    # distr = "dirichlet"
    # alpha = 0.05
    # distr_and_parameter = distr + "-" + str(alpha)
    # laziness = 0.3
    # if laziness is not None:
    #     distr_and_parameter = distr_and_parameter + "-lazy-" + str(laziness)
    # for i in range(3):
    #     setting = RandomExample(sspa_size, distr, laziness=laziness, parameters=[alpha])
    #     f_path = "setting_data/random-size-{}-{}-({})".format(sspa_size, distr_and_parameter, i)
    #     save_bandit(setting, f_path, {"alpha":alpha})
    #     print_bandit(setting)
    #     analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
    #     y = analyzer.solve_lp()[1]
    #     print(y)
    #     W = analyzer.compute_W(abstol=1e-10)[0]
    #     print("W=", W)
    #     print("lambda_W = ", np.linalg.norm(W, ord=2))
    #     U = analyzer.compute_U(abstol=1e-10)[0]
    #     print("U=\n", U)
    #     if U is not np.inf:
    #         print("spectral norm of U=", np.max(np.abs(np.linalg.eigvals(U))))

    sspa_size = 16
    point_distr = "square"
    edge_distr = "uniform"
    threshold = 0.5
    reward_distr = "uniform" #"dirichlet" or "uniform"
    distr_and_parameter = "-".join([point_distr, edge_distr, "thresh="+str(threshold), reward_distr])
    reward_alpha = 0.05
    parameters = {"reward_alpha": reward_alpha}
    if reward_distr == "dirichlet":
        distr_and_parameter += "-" + str(reward_alpha)
    for i in range(7):
        setting = GeometricRandomExample(sspa_size, point_distr, edge_distr, threshold, reward_distr, parameters=parameters)
        f_path = "setting_data/RG-{}-{}-({})".format(sspa_size, distr_and_parameter, i)
        print(f_path)
        setting.visualize_graph()
        if i >=3:
            setting.save(f_path)
        print_bandit(setting)
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
        y = analyzer.solve_lp()[1]
        print(y)
        W = analyzer.compute_W(abstol=1e-10)[0]
        print("W=", W)
        print("lambda_W = ", np.linalg.norm(W, ord=2))
        U = analyzer.compute_U(abstol=1e-10)[0]
        print("U=\n", U)
        if U is not np.inf:
            print("spectral norm of U=", np.max(np.abs(np.linalg.eigvals(U))))



if __name__ == '__main__':
    # generate and save some random settings
    np.random.seed(114514)
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=600)
    np.set_printoptions(suppress=True)
