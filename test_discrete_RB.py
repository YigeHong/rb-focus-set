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


if __name__ == '__main__':
    unittest.main()
