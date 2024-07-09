from rb_settings import *
from discrete_RB import *
from matplotlib import pyplot as plt



def compute_P_and_Phi_eigvals(setting):
    """
    return 1 if unichain and locally stable; return 0 if locally unstable;
    """
    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
    y = analyzer.solve_lp(verbose=False)[1]
    ind_neu = np.where(np.all([y[:,0] > analyzer.EPS, y[:,1] > analyzer.EPS],axis=0))[0]
    if len(ind_neu) > 1:
        return None

    Ppibs_centered = analyzer.Ppibs - np.outer(np.ones((analyzer.sspa_size,)), analyzer.state_probs)
    Ppibs_second_eig = np.max(np.abs(Ppibs_centered))

    Phi = analyzer.compute_Phi(verbose=False)
    Phi_spec_rad = np.max(np.abs(np.linalg.eigvals(Phi)))

    return Ppibs_second_eig, Phi_spec_rad


def count_locally_unstable(alpha):
    sspa_size = 10
    distr = "dirichlet"
    laziness = None
    num_examples = 1000
    unichain_threshold = 0.95
    distr_and_parameter = distr
    if distr == "dirichlet":
        distr_and_parameter += "-" + str(alpha)
    if laziness is not None:
        distr_and_parameter += "-lazy-" + str(laziness)
    unichain_stability_table = np.zeros((2,2))
    eig_vals_list = []
    for i in range(num_examples):
        setting = RandomExample(sspa_size, distr, laziness=laziness, parameters=[alpha])
        result = compute_P_and_Phi_eigvals(setting)
        if result is not None:
            eig_vals = result
            eig_vals_list.append(eig_vals)
            if eig_vals[0] < unichain_threshold:
                is_unichain = 1
            else:
                is_unichain = 0
            if eig_vals[1] < 1:
                is_locally_stable = 1
            else:
                is_locally_stable = 0
            unichain_stability_table[is_unichain, is_locally_stable] +=1
    print(unichain_stability_table)
    unstable_unichain_count = unichain_stability_table[1,0]
    unichain_count = np.sum(unichain_stability_table[1, :])
    eig_vals_list = np.array(eig_vals_list)
    plt.scatter(eig_vals_list[:,0], eig_vals_list[:,1])
    plt.plot([0,1], [1,1], linestyle="--", color="r")
    plt.plot([1,1], [0,1], linestyle="--", color="r")
    plt.xlabel("Second largest modulus of eigenvalues of P_pibs ")
    plt.ylabel("Spectral radius of Phi")
    plt.xlim([0, 1.1])
    plt.ylim([0, 2])
    plt.title("{}, unichain thresh {}, unstable / unichain = {} / {}".format(distr_and_parameter, unichain_threshold, unstable_unichain_count, unichain_count))
    plt.savefig("figs2/eigen-scatter-{}-size-{}.png".format(distr_and_parameter, sspa_size))
    plt.show()


if __name__ == "__main__":
    np.random.seed(114514)
    for alpha in [0.05, 0.1, 0.2, 0.5, 1]:
        count_locally_unstable(alpha)
