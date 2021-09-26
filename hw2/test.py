import numpy as np

class Doge(object):
    gamma = 0.9

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        n = len(rewards)
        list_gamma_prod = [self.gamma ** i for i in range(n)]
        
        A = np.empty(shape=(n, n))
        for i in range(n):
            for j in range(n):
                pow = (n - 1) - (i + j)
                A[i, j] = list_gamma_prod[pow] if pow >= 0 else 0

        r = np.array([rewards[::-1]]).T
        list_of_discounted_cumsums = np.matmul(A, r)
        print(list_of_discounted_cumsums)

        return list_of_discounted_cumsums

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        # curr_sum, curr_gamma_prod = 0, 1
        # list_of_discounted_returns = []
        # for i in range(len(rewards)):
        #     curr_return = curr_sum + curr_gamma_prod * rewards[i]
        #     list_of_discounted_returns.append(curr_return)
        #     curr_sum += curr_return
        #     curr_gamma_prod *= self.gamma
        list_of_discounted_returns = [sum([self.gamma ** t * rewards[t] for t in range(len(rewards))])] * len(rewards)
        print(list_of_discounted_returns)

        return list_of_discounted_returns

rewards = [58, 79, 112]
a = Doge()
a._discounted_cumsum(rewards)
a._discounted_return(rewards)