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


def print_bandit(setting):
    #print("generated using {} distribution".format(self.distr))
    print("state space size = ", setting.sspa_size)
    print("P0 = \n", setting.trans_tensor[:,0,:])
    print("P1 = \n", setting.trans_tensor[:,1,:])
    print("R0 = \n", setting.reward_tensor[:,0])
    print("R1 = \n", setting.reward_tensor[:,1])

def save_bandit(setting, f_path, other_params):
    data_dict = {"sspa_size": setting.sspa_size,
         "trans_tensor": setting.trans_tensor,
         "reward_tensor": setting.reward_tensor,
         "distr": setting.distr
         }
    data_dict.update(other_params)
    with open(f_path, 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    example1 = Gast20Example1()
    print(example1.trans_tensor[0, 1,:])
    print(example1.reward_tensor[1, 1])
