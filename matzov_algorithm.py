import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import logging
import os 
import resource
from random import randint
from fpylll import IntegerMatrix
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
from g6k.utils.stats import dummy_tracer
from math import sqrt, ceil, pi, sin, cos, exp
from fpylll.util import gaussian_heuristic
from itertools import product
from datetime import datetime

# Matzov - column notation, g6k - row notation 

n = 45
m = 40
q = 1489
p = 5

k_enum = 4
k_fft = 4
k_lat = n - k_enum - k_fft

Beta1 = 35
Beta2 = 35
D = 65
'''
n = 30
m = 30
q = 1489
p = 5

k_enum = 2
k_fft = 2
k_lat = n - k_enum - k_fft

Beta1 = 25
Beta2 = 25
D = 33
'''
threads = 6
LWEalpha = 0.015
#sigma = q * LWEalpha

max_sieving = 20

eta = 2
probabilities_B2 = {0: 0.375, 1: 0.25, -1: 0.25, 2: 0.0625, -2: 0.0625}
probabilities_B3 = {0: 0.3125, 1: 0.234375, -1: 0.234375, 2: 0.09375, -2: 0.09375, 3: 0.015625, -3: 0.015625}
entropy_B2 = 2.03 #sigma = 1.0
entropy_B3 = 2.33 #sigma = 1.22

# variables used for plotting
l_lengths = []
l_avgs = []
current_date = None
folder_path = None
variables = {
    'n': n,
    'm': m,
    'q': q,
    'p': p,
    'k_enum': k_enum,
    'k_fft': k_fft,
    'k_lat': k_lat,
    'Beta1': Beta1,
    'Beta2': Beta2,
    'eta': 2
}

'''
g6k (row notation): n x m, n - number of secret coefficients, m - number of samples 
s: m x 1, e: 1 x n 
MATZOV (column notation): m x n, m - number of samples, n - number of coefficients
s: 1 x n , e: m x 1
'''
def gamma_beta(beta):
    return pow(((beta / (2 * math.pi * math.e)) * pow((math.pi * beta), 1. / beta) ), 1. / (beta - 1.))

def asympthotic_D(l_avgs):
    sigma = 1
    mi_values = np.arange(0.05, 1, 0.05)
    D_estimates = []
    #l_sigma = pow(sigma, (m / (m + k_lat))) * pow((sigma * q), (k_lat / (m + k_lat))) * sqrt(4/3) * sqrt(Beta2 / (2 * math.pi * math.e)) * pow(gamma_beta(Beta1), (m + k_lat - Beta2) / 2)
    for l_avg in range(len(l_avgs)):
        D_estimate = []
        for mi in mi_values:
            D_est = (k_enum * entropy_B2 + k_fft * math.log(p) + math.log(1./mi)) * \
                math.exp((k_fft / 3) * pow((sigma * math.pi) / p, 2)) * \
                math.exp((4 * pow(((l_avg * sigma * math.pi) / q),2)))   
            D_estimate.append(D_est)
        D_estimates.append(D_estimate)

    D_estimates = np.mean(D_estimates, axis=0)
    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(mi_values, D_estimates, marker='o')
    plt.title(f'Estimated required number of samples by probability of failure\n(Average form all trials)')
    plt.xlabel('Probability of failure (mi)')
    plt.ylabel('Estimated required number of samples (D)')
    plt.xticks(mi_values) 
    plt.grid(True)
    plt.savefig(folder_path + f"D_estimate_{n}_{m}_{q}" + '.png')

def entropy(probabilities):
    entropy_val = 0
    for prob in probabilities.values():
        entropy_val += prob * math.log2(prob)
    return -entropy_val

def log2(x):
    return int(math.log(x) // math.log(2))

def vector_length(vector):
    return math.sqrt(sum(v_**2 for v_ in vector))

def binomial(eta):
    """
    Compute the Binomial function.

    Parameters:
    - eta (int): Number of iterations.

    Returns:
    - int: The result of the Binomial function.
    """
    sum_result = 0

    for i in range(eta):
        a = random.randint(0, 1)
        b = random.randint(0, 1)
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

    def __init__(self, sigma, tau=10):
        """
        Discrete Gaussian Sampler over the integers

        :param sigma: targeted standard deviation (std.dev. of the discrete
        gaussian is close to sigma, but not necessarily equal).
        :param tau: number of standard deviations away at which we will perform
        a tail-cut, i.e. there will not be any value outputted that is larger
        than tau * sigma in absolute value.

        :returns: DiscreteGaussian object
        """
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

def progressive_sieve_left(g6k):
    r = g6k.r
    #gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(g6k.r - Beta2, g6k.r)])
    gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(r)])
    print(gh)
    #print(gaussian_heuristic([g6k.M.get_r(i, i) for i in range(Beta2)]))
    g6k.initialize_local(0, max(0, r - 20), r)
    print(f"m + k_lat: {m + k_lat}")
    print(f"r in G6K: {g6k.r}")
    print(f"l in G6K: {g6k.l}")
    while g6k.r - g6k.l < min(Beta2, 45):
        g6k.extend_left(1)
        g6k("gauss")
        print(g6k.l)
        logging.info(f"Gauss sieving with size: {g6k.l}")

    while g6k.r - g6k.l < Beta2:
        g6k.extend_left(1)
        g6k("bgj1")
        print(g6k.l)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024  # 1 KB = 1024 B
        print("Memory usage (MB):", memory_mb)
        logging.info(f"BGJ1 sieving with size: {g6k.l}")

    print(f"Sieving with hk3")
    with g6k.temp_params(saturation_ratio=.9, db_size_factor=3):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024  # 1 KB = 1024 B
        print("Memory usage (MB):", memory_mb)
        g6k(alg="hk3")

    #g6k.resize_db(ceil(.9 * (4 / 3)**((Beta2) / 2)))

    db = list(g6k.itervalues())
    database = []
    
    print(f"Database size: {len(db)}")
    logging.info(f"Database size:: {len(db)}")

    for x in db:
        #v = g6k.M.B[g6k.r - Beta2:].multiply_left(x)
        v = g6k.M.B[g6k.l: g6k.r].multiply_left(x)
        #v = g6k.M.B.multiply_left(x)
        l = sum(v_**2 for v_ in v)                
        if l < 10 * gh:  
            database.append(v) 
            #print((l/gh, v))  
        
    print(f"Final database size: {len(database)}")
    logging.info(f"Final database size:: {len(database)}")

    return [w[:m] for w in database] + [[-x for x in w[:m]] for w in database]


def progressive_sieve_right(g6k):

    gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(Beta2)])
    g6k.initialize_local(0, 0, 0)
    #saturation_radius = 1.3

    while g6k.r < min(Beta2, 45):
        g6k.extend_right(1)
        g6k("gauss")
        print(g6k.r)
        logging.info(f"Gauss sieving with size: {g6k.r}")

    while g6k.r < Beta2:
        with g6k.temp_params(saturation_ratio=0.001, saturation_radius = 1.3, db_size_factor=10):
            g6k.extend_right(1)
            g6k("bgj1")
            print(g6k.r)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = usage.ru_maxrss / 1024  # 1 KB = 1024 B
            print("Memory usage (MB):", memory_mb)
            logging.info(f"BGJ1 sieving with size: {g6k.r}")

    db = list(g6k.itervalues())
    database = []

    print(f"Database size: {len(db)}")
    logging.info(f"Database size:: {len(db)}")
    smaller = 0
    for x in db:
        v = g6k.M.B.multiply_left(x)
        l = sum(v_**2 for v_ in v)                
        if l < 1.7 * gh:  
            database.append(v) 
            #print((l/gh, v))    

    print(f"Smaller: {smaller}")
    print(f"Final database size: {len(database)}")
    logging.info(f"Final database size:: {len(database)}")

    return [w[:m] for w in database] + [[-x for x in w[:m]] for w in database]

def generate_permutations(eta, n):

    if eta == 2:
        numbers = [-2, -1, 0, 1, 2]
        probabilities = probabilities_B2
    else:
        numbers = [-3, -2, -1, 0, 1, 2, 3]
        probabilities = probabilities_B3

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

def mod_switch_distinguisher(y_enums, y_ffts, q, p, k_fft,  L, s_tilde_enum, b):

    table = np.zeros(shape=(p,) * k_fft, dtype=complex)

    for (j, x_j) in enumerate(L):
        index = tuple(int(round(1. * p * x / q) % p) for x in y_ffts[j])
        inner_product = np.inner(x_j, b) - np.inner(y_enums[j], s_tilde_enum)
        angle = inner_product * 2 * pi / q
        table[index] += cos(angle) + sin(angle) * 1.j

    fft_output = np.fft.fftn(table).real
    return np.amax(fft_output)   

# row notation
def generate_LWE_instance(m, n, q):
    """
    Generate an LWE instance

    Parameters:
    - m (int): Number of samples.
    - n (int): Number of secret coefficients.
    - q (int): Modulo.

    Returns:
    - A (IntegerMatrix): LWE matrix of size n x m, where n is the number of secret coefficients and m is the number of samples
    - b (list): Target matrix of size 1 x m
    - s (list): Secret matrix of size 1 x n
    """
    # Generate a random q-ary lattice with
    #B = IntegerMatrix.random(m + n, 'qary', k = m, q = q)
    B = IntegerMatrix.random(m + n, 'qary', k = m, q = q)
    '''
        print(IntegerMatrix.random(10, "qary", k=8, q=127))
        [ 1 0  50  44   5   3  78   3  94  97 ]
        [ 0 1  69  12 114  43 118  47  53   4 ]
        [ 0 0 127   0   0   0   0   0   0   0 ]
        [ 0 0   0 127   0   0   0   0   0   0 ]
        [ 0 0   0   0 127   0   0   0   0   0 ]
        [ 0 0   0   0   0 127   0   0   0   0 ]
        [ 0 0   0   0   0   0 127   0   0   0 ]
        [ 0 0   0   0   0   0   0 127   0   0 ]
        [ 0 0   0   0   0   0   0   0 127   0 ]
        [ 0 0   0   0   0   0   0   0   0 127 ]
    '''
    A = B.submatrix(0, n, n, m + n)

    s = list()
    e = list()

    for _ in range(n):
        s.append(binomial(eta))

    for _ in range(m):    
        e.append(binomial(eta))

    b = A.multiply_left(s) # return s * A
    b = [(b[i] + e[i]) % q for i in range(m)]

    return A, b, s

def create_dual_lattice(A_lat):
    """
    Create a dual lattice froom lattice A_lat

    Parameters:
    - A_lat (IntegerMatrix): Matrix of size m x k_lat

    Returns:
    - B_dual (IntegerMatrix): Dual Lattice of size (m + k_lat) x (m + k_lat) and type (in row notation):
                            [I_m, A_lat]
                            [0, q * I_kat]
    """
    B_dual = IntegerMatrix.identity(m + k_lat)
    for i in range(m, m + k_lat):
        B_dual[i, i] *= q
    for i in range(0, m):
        for j in range(0, k_lat):
            B_dual[i, m + j] = pow(A_lat[i, j], 1, q)

    return B_dual

def sampling(B_dual):
    global l_lengths
    sieverParams = SieverParams(threads = threads, dual_mode = False)
    g6k = Siever(M = B_dual, params=sieverParams)
    short_vectors = []
    l_length = []
    for _ in range(max_sieving):
        bkz(g6k, dummy_tracer, Beta1) # BKZ reduction with block size Beta1'''   
        logging.info(f"BKZ reduction completed")
        new_short_vectors = progressive_sieve_left(g6k)
        for vector in new_short_vectors:
            if vector not in short_vectors:
                short_vectors.append(vector)
        logging.info(f"New vectors sampled, number of vectors: {len(short_vectors)}")
        if len(short_vectors) >= D:
            short_vectors = sorted(short_vectors, key=lambda x: vector_length(x))
            short_vectors = short_vectors[:D]
            l_length = [vector_length(vector) for vector in short_vectors]
            break
    l_lengths.append(l_length)
    print(l_lengths)
    l_avgs.append(sum(l_length) / len(l_length))
    print("Final DB size: ")
    print(len(short_vectors))
    print(f"Average lengh l: {sum(l_length) / len(l_length)}")
    logging.info(f"Vectors used\n: {short_vectors}")
    logging.info(f"Average lengh l: {sum(l_length) / len(l_length)}")
    #asympthotic_D(sum(l_length) / len(l_length))
    return short_vectors

def dual_attack_full():
    global k_lat, k_enum, k_fft
    assert k_enum + k_fft + k_lat == n
    A, b, s = generate_LWE_instance(m, n, q)
    s_guess = []
    
    while k_lat >= k_enum + k_fft:
        s_guess.append(dual_attack(A, b, s))
        A = A[k_enum:]
        s = s[k_enum:]
        k_lat -= k_enum
        k_enum = 1

def dual_attack(A, b, s):

    #Line 1: 
    A_enum = A[:k_enum]
    A_fft = A[k_enum: k_enum + k_fft]
    A_lat = A[k_enum + k_fft:]

    A_enum.transpose()
    A_fft.transpose()
    A_lat.transpose()

    #Line 2 : 
    B_dual = create_dual_lattice(A_lat)

    #Line 3 : 
    L = sampling(B_dual)

    y_ffts = []
    y_enums = []

    for i in range(0, len(L)):
        v = L[i]
        y_ffts.append(tuple(element % q for element in A_fft.multiply_left(v)))       
        y_enums.append(tuple(element % q for element in A_enum.multiply_left(v)))

    # Line 4:
    s_tilde_enums = generate_permutations(2, k_enum)
    list_max = []
    probs = []
    for i in range(len(s_tilde_enums)):
        _max = mod_switch_distinguisher(y_enums, y_ffts, q, p, k_fft, L, s_tilde_enums[i], b)
        probs.append((_max, s_tilde_enums[i]))
        list_max.append(_max)

    index = np.argmax(list_max)

    return s_tilde_enums[index]

def dual_attack_test(A, b, s):

    #Line 1: 
    A_enum = A[:k_enum]
    A_fft = A[k_enum: k_enum + k_fft]
    A_lat = A[k_enum + k_fft:]

    A_enum.transpose()
    A_fft.transpose()
    A_lat.transpose()

    #Line 2 : 
    B_dual = create_dual_lattice(A_lat)
    
    #Line 3 : 
    start_time = time.time()
    L = sampling(B_dual)
    L = L[:D]
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    logging.info(f"Sampling completed, execution time: {execution_time}")

    d = len(L)
    correct = [0] * log2(d)
    y_ffts = []
    y_enums = []
    for i in range(0, d):
        v = L[i]
        y_ffts.append(tuple(element % q for element in A_fft.multiply_left(v)))       
        y_enums.append(tuple(element % q for element in A_enum.multiply_left(v)))
        
    new_k_enum = k_enum
    # Line 4:
    s_tilde_enums = generate_permutations(eta, new_k_enum)
    for exponent in range(0, log2(d)):
        power_of_two = 2 ** (exponent + 1)
        
        probs = []
        for i in range(len(s_tilde_enums)):
            _max = mod_switch_distinguisher(y_enums, y_ffts, q, p, k_fft, L[:power_of_two], s_tilde_enums[i], b)
            probs.append((_max, s_tilde_enums[i]))
            #list_max.append(_max)
        # get the rank
        matching_tuple = next(item for item in probs if np.array_equal(item[1], s[:new_k_enum]))
        # Posortowana tablica po pierwszej wartości w każdej tupli
        sorted_array = sorted(probs , key=lambda x: x[0])
        index2 = sorted_array.index(matching_tuple)
        index = len(sorted_array) - 1

        print("exponent: ")
        print(exponent)
        print("guess: ")
        print(sorted_array[index][1])
        print("correct: ")
        print(s[:new_k_enum])
        logging.info(f"Rank for exponent {exponent} : {abs(index2 - index)}")
        correct[exponent] = abs(index2 - index)

    return correct


def test_single_guess():

    assert k_enum + k_fft + k_lat == n
    correct = 0
    trials = 1

    for i in range(0, trials):
        A, b, s = generate_LWE_instance(m, n, q)
        s_guess =  dual_attack(A, b, s)
        print(int(np.array_equal(s_guess, s[:k_enum])))
        correct += int(np.array_equal(s_guess, s[:k_enum]))

    print("Correct Guesses (%)")
    print(correct)
    print(trials)
    print(correct / trials)

def test_vectors():
    global current_date 
    global folder_path
    global l_lengths 

    current_date = datetime.now().strftime("%m-%d-%H-%M")
    folder_path = "d_test/" + f"{current_date}_{n}_{m}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_name = folder_path + "_logs.txt"
    logging.basicConfig(filename=log_name, level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    assert k_enum + k_fft + k_lat == n

    A, b, s = generate_LWE_instance(m, n, q)

    variables['A'] = A
    variables['b'] = b
    variables['s'] = s

    
    file_name = f"variables_{n}_{m}.txt"

    # Create the file in the folder
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        for variable, value in variables.items():
            file.write(f'{variable}:{value}\n')

    num_of_trials = 1
    logging.info(f"Number of trials: {num_of_trials}")
    s_guess = []
    for _ in range (num_of_trials):
        s_guess.append(dual_attack_test(A, b, s))

    s_guess = np.mean(s_guess, axis=0)
    
    asympthotic_D(l_avgs)
    # Create a box plot
    print(l_lengths)
    temp = len(l_lengths)
    sums = [sum(column) for column in zip(*l_lengths)]
    averages = [sum_value / temp for sum_value in sums]

    print(log2(len(averages)))
    # Define the ranges for boxplot series
    ranges = [2**i for i in range(1, log2(len(averages)) + 1)]

    # Prepare data for boxplots
    data = [averages[:r] for r in ranges]
    print(ranges)
    print(len(ranges))
    print(len(averages))
    # Create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data)
    plt.xticks(np.arange(1, len(ranges) + 1), [f'2^{i + 1}' for i in range(len(ranges))])
    plt.xlabel('Number of Vectors')
    plt.ylabel('Vectors length')
    plt.title(f'Length of vectors used for attack\n(n={n}, m={m}, k_fft={k_fft}, k_enum={k_enum})\naverage of {num_of_trials} trials')
    plt.grid(True)
    plt.show()
    plot_name = f"box_{n}_{m}"
    plt.savefig(folder_path + plot_name + '\n(n={n}, m={m}, Beta1={Beta1}, Beta2={Beta2}' + '.png')

    plot_title = f"Effect of the number of vectors used in FFT distinguisher on the rank of correct s_enum\n(n={n}, m={m}, k_fft={k_fft}, k_enum={k_enum})\naverage of {num_of_trials} trials"
    plot_name = f"D_{n}_{m}"
    x_values = list(range(1, len(s_guess) + 1)) 
    x_labels = [f"2^{x}" for x in x_values]  # Converting to powers of two

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.scatter(x_values, s_guess)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.title(plot_title) 
    plt.xlabel('Number of vectors')  
    plt.ylabel('Rank of correct s_enum') 
    plt.xticks(x_values, x_labels)
    plt.grid(True)
    plt.savefig(folder_path + plot_name + '.png')

#dual_attack_full()
#test_single_guess()
test_vectors()