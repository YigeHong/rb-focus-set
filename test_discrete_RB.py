import unittest
import rb_settings
from discrete_RB import *
from experiments import test_cycle


class TestDiscreteRB(unittest.TestCase):
    def test_solver(self):
        settings = [rb_settings.Gast20Example1(), rb_settings.Gast20Example2(), rb_settings.Gast20Example3()]
        true_opt_values = [0.2476, 0.1238, 0.3638] #[0.2476, 0, 0]
        for i, setting in enumerate(settings):
            setting = settings[i]
            true_opt_value = true_opt_values[i]
            act_frac = 0.4
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=act_frac)
            solved_opt_value, opt_var = analyzer.solve_lp()
            self.assertAlmostEqual(solved_opt_value, true_opt_value, places=3)
            print("Example {} solved correctly".format(i+1))

    def test_resolving(self):
        """
        test whether modifying the transition and reward will affect the correctness of the solution.
        """
        settings = [rb_settings.Gast20Example1(), rb_settings.Gast20Example2(), rb_settings.Gast20Example3()]
        true_opt_values = [0.2476, 0.1238, 0.3638] #[0.2476, 0, 0]
        for i, setting in enumerate(settings):
            setting = settings[i]
            true_opt_value = true_opt_values[i]
            act_frac = 0.4
            if i == 0:
                analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=act_frac)
            else:
                analyzer.trans_tensor = setting.trans_tensor
                analyzer.reward_tensor = setting.reward_tensor
            solved_opt_value, opt_var = analyzer.solve_lp()
            self.assertAlmostEqual(solved_opt_value, true_opt_value, places=3)
            print("Example {} solved correctly".format(i+1))

    def test_indexable_cycle_finder(self):
        for i, setting in enumerate([rb_settings.Gast20Example1(),
                        rb_settings.Gast20Example2(),
                        rb_settings.Gast20Example3()]):
            setting = rb_settings.Gast20Example1() #rb_settings.RandomExample(sspa_size, distr="beta05")
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
            priority_list, indexable = analyzer.solve_whittles_policy()
            has_cycle = test_cycle(setting, priority_list, act_frac=0.4, try_steps=100, eps=1e-4)
            self.assertTrue(indexable)
            self.assertTrue(has_cycle)
            print("Test on example {} passed".format(i+1))

    def test_equivalence_of_wip_lp(self):
        """
        wip and lp are known to be equal on some examples
        """
        for i, setting in enumerate([rb_settings.Gast20Example1(), rb_settings.Gast20Example2(), rb_settings.Gast20Example3()]):
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
            whittle_priority_list, indexable = analyzer.solve_whittles_policy()
            lp_priority_list = analyzer.solve_LP_Priority()
            self.assertEqual(lp_priority_list, whittle_priority_list)
            # WIP and LP-priority are the same on the three examples in Gast et.al. 2020
            print("WIP and LP-priority are the same on Example {}".format(i))


    def test_sanity_of_solving_q_function(self):
        """
        # sanity check of the q-function when solving for the LP-priority
        """
        for i, setting in enumerate([rb_settings.Gast20Example1(), rb_settings.Gast20Example2(), rb_settings.Gast20Example3()]):
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
            analyzer.solve_LP_Priority()
            self.assertTrue(analyzer.avg_reward > 0)
            self.assertTrue(analyzer.opt_subsidy)
            action_gap = analyzer.q_func_relaxed[:,1] - analyzer.q_func_relaxed[:,0]
            # find out the fluid active and passive classes
            fluid_active = np.all([analyzer.y.value[:, 0] < 1e-4, analyzer.y.value[:, 1] > 1e-4], axis=0)
            fluid_passive = np.all([analyzer.y.value[:, 0] > 1e-4, analyzer.y.value[:, 1] < 1e-4], axis=0)
            # action gap is positive if only if the state is fluid active; negative if only if the state is fluid passive
            self.assertTrue(np.all((action_gap > 1e-4)==fluid_active),
                            msg="action_gap\n {} \n, y=\n {}".format(action_gap, analyzer.y.value))
            self.assertTrue(np.all((action_gap < - 1e-4)==fluid_passive),
                            msg="action_gap\n {} \n, y=\n {}".format(action_gap, analyzer.y.value))



if __name__ == '__main__':
    unittest.main()
