import numpy as np
import pickle


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
    def __init__(self, sspa_size, distr, verbose=False):
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
        else:
            raise NotImplementedError
        P0 = P0 / np.sum(P0, axis=1, keepdims=True)
        P1 = P1 / np.sum(P1, axis=1, keepdims=True)
        self.trans_tensor = np.stack([P0, P1], axis=1) # dimensions (state, action, next_state)
        self.reward_tensor = np.stack([R0, R1], axis=1) # dimensions (state, action)
        # make sure alpha is not too close to 0 or 1
        self.suggest_act_frac = np.random.uniform(0.1,0.9)

        if verbose:
            print("P0 = ", P0)
            print("P1 = ", P1)
            print("R0 = ", R0)
            print("R1 = ", R1)




class ExampleFromFile(object):
    def __init__(self, f_path):
        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.sspa_size = data_dict["sspa_size"]
        self.trans_tensor = data_dict["trans_tensor"]
        self.reward_tensor = data_dict["reward_tensor"]
        self.distr = data_dict["distr"]
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
    def __init__(self, sspa_size, probs_L, probs_R, action_script, suggest_act_frac):
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

    @staticmethod
    def get_parameters(name, sspa_size):
        """
        :param name:
        :param sspa_size: size of the mdp
        :return: probs_L, probs_R
        """
        assert sspa_size % 2 == 0, "state-space size should be an even number"
        half_size = int(sspa_size / 2)
        if name == "eg4unif-tb":
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
        elif name == "eg4action-gap-tb":
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
        elif name == "eg4archive1":
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
        else:
            raise NotImplementedError
        # produce the suggested activation fraction. It should be just enough to let the arm always move to the right
        tries_in_a_loop = 1 / probs_R
        suggest_act_frac = np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
        return probs_L, probs_R, action_script, suggest_act_frac


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



def print_bandit(setting):
    #print("generated using {} distribution".format(self.distr))
    print("state space size = ", setting.sspa_size)
    print("P0 = \n", setting.trans_tensor[:,0,:])
    print("P1 = \n", setting.trans_tensor[:,1,:])
    print("R0 = \n", setting.reward_tensor[:,0])
    print("R1 = \n", setting.reward_tensor[:,1])
    if hasattr(setting, "suggest_act_frac"):
        print("suggest act frac = ", setting.suggest_act_frac)


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


if __name__ == '__main__':
    # generate and save some random settings
    np.random.seed(114514)
    sspa_size = 4
    distr = "uniform"
    for i in range(3):
        setting = RandomExample(sspa_size, distr)
        f_path = "setting_data/random-size-{}-{}-({})".format(sspa_size, distr, i)
        save_bandit(setting, f_path, None)
