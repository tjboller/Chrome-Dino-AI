import numpy as np
import AIs
import pickle
import run

# SPSA - Simultaneous perturbation stochastic approx
def spsa(f, theta, file_name = 'spsa_export',alpha=0.602, gamma=.101, a=5, big_a=None, c=20, max_iter=1000, report=10):
    """
    http://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF
    :param f: function to minimize
    :param theta: initial input
    :param alpha: Parameter - Optimal shown to be .101
    :param gamma: Parameter - Optimal shown to be .602
    :param big_a: Less than 10% of max_iterations
    :param a: a/(A+1)^alpha about equal to the smallest desired change magnitudes
    :param c: Set c at the level approximately equal to the st dev can be est. Does not need to be precise
    :param max_iter: Max number of iterations allowed
    :param report: How often you want to see printed report
    :return: optimal theta
    """

    if big_a is None:
        big_a = (max_iter * .05)

    n_iter = 0
    scores = []
    thetas = []

    while n_iter < max_iter:

        a_n = a/(n_iter + 1 + big_a)**alpha
        c_n = c/(n_iter + 1)**gamma

        perturbation = np.random.choice([1, -1], theta.shape)

        theta_plus = theta + c_n * perturbation
        theta_minus = theta - c_n * perturbation

        cost_plus = f(theta_plus)
        cost_minus = f(theta_minus)

        # - cost is specific to this application
        thetas.append(theta)

        grad = (cost_plus - cost_minus)/(2 * c_n * perturbation)
        theta = theta - a_n * grad

        if report > 0 and n_iter > 0 and n_iter % report == 0:

            score = f(theta)
            scores.append(score)

            print('Run {0}, Score of new strategy: {2}'
                  .format(n_iter, report, score))
            print('Current theta {0}'.format(theta))

            pickle.dump((theta, scores, thetas), open(file_name + ".p", "wb" ))

        n_iter += 1

    return theta, scores, thetas


if __name__ == "__main__":

    # get cost curve
    # rule_scores = []
    # for rule in range(60, 81):
    #     ai = AIs.RuleBased(rule)
    #     for run in range(20):
    #         score = -ai.get_performance_cost(num_runs=1)
    #         rule_scores.append((rule,score))
    #     print(rule_scores)
    #     pickle.dump(rule_scores, open("rule_cost_curve.p", "wb"))

    # starting from bottom
    # ai = AIs.RuleBased(60)
    # result = spsa(ai.get_performance_cost, ai.strategy, file_name='rule_based_60')

    # check if same answer from top
    # ai = AIs.RuleBased(80)
    # result = spsa(ai.get_performance_cost, ai.strategy, file_name='rule_based_80')

    # logistic AI, start with best rule
    # hand curated that works well [95, -1, -.7, 2]
    ai = AIs.Logistic([74.8, -.8, -.95, .95])
    # result = spsa(ai.get_performance_cost, ai.strategy, file_name='logistic',
    #              c=1, a=.5)

    run.run(ai=ai)
