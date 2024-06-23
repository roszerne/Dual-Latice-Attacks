import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binom, norm
from random import randint, random
from math import pi, exp, log, log2, comb
from itertools import product

def entropy(probabilities):
    """
    Calculate the entropy value based on the given probabilities.

    :param probabilities: A dictionary containing probabilities.

    :return: float: The entropy value calculated from the given probabilities.

    This function calculates the entropy value using the given probabilities. It iterates through the values
    of the probabilities dictionary, computing the entropy contribution for each probability, and then sums up
    all the entropy contributions. Finally, it returns the negative sum of these entropy contributions.
    """
    entropy_val = 0
    for prob in probabilities.values():
        entropy_val += prob * log2(prob)
    return -entropy_val

class BinomialDistribution():
#   (n, p) = (2*eta, 0.5)
    def __init__(self, eta):
        if eta == 2:
            self.probability_distribution = {0: 0.375, 1: 0.25, -1: 0.25, 2: 0.0625, -2: 0.0625}
            self.entropy = 2.03
            self.sigma = 1.0
            self.eta = eta
        else:
            self.probability_distribution = {0: 0.3125, 1: 0.234375, -1: 0.234375, 2: 0.09375, -2: 0.09375, 3: 0.015625, -3: 0.015625}
            self.entropy = 2.33
            self.sigma = 1.22
            self.eta = eta

    def PDF(self, outcome):
        return 0.25**(self.eta) * comb(2*self.eta, outcome + self.eta)

    def CDF(self, x):
        return binom.cdf(x, 2 * self.eta, 0.5)

    def CDF_INV(self, quantile):
        if quantile >= 1:
            quantile = 0.9999
        #initial_guess = eta / 2
        initial_guess = 0
        cdf_value = binom.cdf(initial_guess, 2 * self.eta, 0.5)
        while cdf_value < quantile:
            initial_guess += 1
            cdf_value = binom.cdf(initial_guess, 2 * self.eta, 0.5)
        return initial_guess

    def generate_permutations(self, n):
        """
        Generate permutations of numbers with descending order of probability based on the value of eta and the length of the permutation.

        Parameters:

            n (int): Length of each permutation.

        Returns:
            numpy.ndarray: An array containing all possible permutations of numbers with their respective probabilities,
                        sorted in descending order of probability sums.
        """
        numbers = self.probability_distribution.keys()
        probabilities = self.probability_distribution

        # Generate all possible combinations of numbers with repetition
        all_combinations = product(numbers, repeat=n)

        # Create a list of tuples where each tuple contains the combination and its probability sum
        lists_with_probabilities = []
        for combination in all_combinations:
            probability_sum = np.sum([probabilities[num] for num in combination])
            lists_with_probabilities.append((combination, probability_sum))

        # Sort the lists based on the probability sum
        sorted_lists = sorted(lists_with_probabilities, key=lambda x: x[1], reverse=True)

        # Extract the combinations from the sorted list
        combinations = [list_with_prob[0] for list_with_prob in sorted_lists]

        # Create a NumPy array from the combinations
        return np.array(combinations)

    def n_enum(self, s):
        """
        Find the number of guesses s_enum_tilde with probabilities larger than the probability of correct s_enum, according to the distribution.

        Parameters:
            s (list): Secret Vector

        Returns:
            n_enum (int): the number of guesses with probabilities larger than the probability of correct s_enum.
        """
        permutations = self.generate_permutations(len(s))
        return np.where((permutations == s).all(axis=1))[0][0]

    def __call__(self):
        """
        Compute the Binomial function.

        Parameters:
        - eta (int): Number of iterations.

        Returns:
        - int: The result of the Binomial function.
        """
        sum_result = 0

        for _ in range(self.eta):
            a = randint(0, 1)
            b = randint(0, 1)
            sum_result += a - b

        return sum_result

class DiscreteGaussian:
    """
    Original code obtained from: github.com/ludopulles/DoesDualSieveWork
    File: MATZOV.py
    Author: ludopulles
    License: MIT 

    Sampler for an integer that is distributed by a discrete gaussian of a
    given standard deviation
    """

    def __init__(self, sigma, tau=2):
        """
        Discrete Gaussian Sampler over the integers

        :param sigma: targeted standard deviation (std.dev. of the discrete
        gaussian is close to sigma, but not necessarily equal).
        :param tau: number of standard deviations away at which we will perform
        a tail-cut, i.e. there will not be any value outputted that is larger
        than tau * sigma in absolute value.

        :returns: DiscreteGaussian object
        """
        self.sigma = sigma
        self.probability_distribution = {}
        self.entropy = None

        # The point at which the gaussian's tail is cut:
        self.tail = int(tau * sigma + 2)
        # Create the cumulative density table of length `tail`, where the i'th
        # entry contains `\sum_{|x| <= i} rho_{sigma}(x)`.
        self.cdt = self.tail * [0]

        factor = -0.5 / (sigma * sigma)
        cum_prod = 1.0
        self.cdt[0] = 1.0
        for i in range(1, self.tail):
            # Exploit the symmetry of P(X = x) = P(X = -x) for non-zero x.
            cum_prod += 2 * exp(factor * (i * i))
            self.cdt[i] = cum_prod
        # The total gaussian weight:
        self.renorm = cum_prod

        for i in self.support():
            self.probability_distribution[i] = self.PDF(i) / 2
        print(f"Dist: {self.probability_distribution}")
        self.entropy = entropy(self.probability_distribution)
        print(f"Entropy: {self.entropy}")

    def support(self):
        """
        Give the range [l, r] on which the PDF is nonzero.
        """
        return range(1 - self.tail, self.tail)

    def PDF(self, outcome):
        """
        Give the probability on a certain outcome
        """
        if outcome == 0:
            return 1.0 / self.renorm
        return (self.cdt[abs(outcome)] - self.cdt[abs(outcome) - 1]) / self.renorm

    def CDF_INV(self, p):
        if p >= 1:
            p = 0.99
        # Step 1: Convert probability to z-score using the standard normal distribution
        z_score = norm.ppf(p)
        
        # Step 2: Convert z-score to the original scale of the distribution
        x = 0 + self.sigma * z_score
        
        return np.real(x)

    def generate_permutations(self, n):
        """
        Generate permutations of numbers with descending order of probability based on the value of eta and the length of the permutation.

        Parameters:

            n (int): Length of each permutation.

        Returns:
            numpy.ndarray: An array containing all possible permutations of numbers with their respective probabilities,
                        sorted in descending order of probability sums.
        """
        numbers = self.probability_distribution.keys()
        probabilities = self.probability_distribution

        # Generate all possible combinations of numbers with repetition
        all_combinations = product(numbers, repeat=n)

        # Create a list of tuples where each tuple contains the combination and its probability sum
        lists_with_probabilities = []
        for combination in all_combinations:
            probability_sum = np.sum([probabilities[num] for num in combination])
            lists_with_probabilities.append((combination, probability_sum))

        # Sort the lists based on the probability sum
        sorted_lists = sorted(lists_with_probabilities, key=lambda x: x[1], reverse=True)

        # Extract the combinations from the sorted list
        combinations = [list_with_prob[0] for list_with_prob in sorted_lists]

        # Create a NumPy array from the combinations
        return np.array(combinations)

    def n_enum(self, s):
        """
        Find the number of guesses s_enum_tilde with probabilities larger than the probability of correct s_enum, according to the distribution.

        Parameters:
            s (list): Secret Vector

        Returns:
            n_enum (int): the number of guesses with probabilities larger than the probability of correct s_enum.
        """
        permutations = self.generate_permutations(len(s))
        return np.where((permutations == s).all(axis=1))[0][0]



    def __call__(self):
        """
        Takes one sample from the Discrete Gaussian

        :returns: the integer that is the output of the sample, i.e. outputs a
        number `x` with probability exp(-x^2 / 2sigma^2)
        """
        rand = random() * self.renorm
        for i in range(self.tail):
            if rand < self.cdt[i]:
                # The probability to end up here is precisely:
                #     (self.cdt[i] - self.cdt[i-1]) / self.renorm = P(|X| = i)
                # Thus, flip a coin to choose the sign (no effect when i = 0)
                return i * (-1)**randint(0, 1)
        # This should not happen:
        return self.tail * (-1)**randint(0, 1)
    