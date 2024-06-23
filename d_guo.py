"""
This script contains the code for the experiments discussed in Chapter 5, Section 1.2.
It examines the bound on the minimal required number of short vectors for the dual attack
proposed by Guo and Johansson.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import multiprocessing
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt, ceil, pi, sin, cos, exp, log, log2
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
from g6k.utils.stats import dummy_tracer
from fpylll import IntegerMatrix
from datetime import datetime
from distributions import BinomialDistribution, DiscreteGaussian
from utils import vector_length, print_memory_usage, multiply_left

# LWE parameters
n = None
m = None
q = None
# attack parameters
p = None
t_1 = None
t = None
k_lat = None
Beta1 = None
Beta2 = None
D = None
C = None
# sieving parameters
saturation_radius = None
saturation_ratio = None
db_size_factor = None
max_sieving = 20
dist = None 
gauss = None

# variables used for plotting
ell = None# List of size num_of_trials, containing average size of D vectors for each trial 
current_date = None # Current date used for creating folder name 
folder_path = None # Path where the result of experiments are stored
sub_trials = None
mi_values = np.arange(0.01, 1, 0.01)
mi = 0.001
variables = None

'''
g6k (row notation): n x m, n - number of secret coefficients, m - number of samples 
s: m x 1, e: 1 x n 
MATZOV (column notation): m x n, m - number of samples, n - number of coefficients
s: 1 x n , e: m x 1
'''

def required_number_of_samples(B, ell, sigma = 1):
    """
    Calculate the required number of samples for the algorithm as given in Subsection 4.3.3

    Parameters:
    B (int): Number of oossible values of the secret distribution 
    ell (float): average length of short vector
    sigma (float, optional): Standard deviation, default is 1

    Returns:
    None: This function prints and logs the required number of samples.
    """
    tau = (sigma* ell) / q
    eps = exp(-2 * pow(pi, 2) * pow(tau, 2))
    c0 = 4 #per 4.1 Complexity Analysis
    N = c0 * log(pow(p, t) * pow(B, t_1)) / pow(eps, 2)
    print(f"Required number of samples: {N}")
    logging.info(f"Required number of samples: {N}")

def progressive_sieve_left(g6k):
    """
    Perform a progressive sieving with extend_left() function

    Parameters:
    g6k (Siever): g6k instance used for sieving

    Returns:
    database (list): List containing .5 * saturation_ratio *saturation_radius**(Beta2/2.) shortest vectors.

    """
    r = g6k.r
    g6k.initialize_local(0, max(0, r - 20), r)
    while g6k.r - g6k.l < min(Beta2, 45):
        g6k.extend_left(1)
        g6k("gauss")
        print(f"Gauss sieving with size: {g6k.l}")
        logging.info(f"Gauss sieving with size: {g6k.l}")

    while g6k.r - g6k.l < Beta2:
        g6k.extend_left(1)
        g6k("bgj1")      
        print_memory_usage()
        print(f"BGJ1 sieving with size: {g6k.l}")
        logging.info(f"BGJ1 sieving with size: {g6k.l}")
    
    print("hk3 sieving with size")
    logging.info("hk3 sieving with size")
    with g6k.temp_params(saturation_ratio = saturation_ratio, saturation_radius = saturation_radius, db_size_base=sqrt(saturation_radius) ,db_size_factor = db_size_factor):
        g6k(alg="hk3")
    print("Finished hk3")
    g6k.resize_db(ceil(saturation_ratio * saturation_radius ** (Beta2/2.)))
    print("Finished resizing")
    while g6k.l > 0:
        print_memory_usage()
        # Extend the lift context to the left
        g6k.extend_left(1)
        print(f"Extending: {g6k.l}")
        logging.info(f"Extending: {g6k.l}")
    print("Finished extending")
    db = list(g6k.itervalues())
    print("Finished listing")
    database = [None] * len(db)
    multiply_left_partial = partial(multiply_left, A = g6k.M.B[g6k.l:g6k.r])
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, result in zip(range(len(db)), executor.map(multiply_left_partial, db)):
            database[i] = result

    print(f"Final database size: {len(database)}")
    logging.info(f"Final database size:: {len(database)}")

    return [w for w in database] 

def distinguisher(y_enums, y_ffts, q, gamma, t, L, s_tilde_enums, b):
    """
    Implementation of the distinguisher of Guo and Johansson

    Parameters:
    y_enums (list): List of short vectors
    y_ffts (list): List of short vectors
    q (int): Modulus of LWE instance
    gamma (int): Modulus of FFT distinguisher
    t (int): parameter of the algorithm
    L (list): List of short vectors from the lattice
    s_tilde_enums (list): current guess on s_enums
    b (list): b vector from an LEW instance

    Returns:
    np.amax(fft_output) (float): The maximum of the real part of Ts.

    """
    table = np.zeros(shape=(gamma,) * t, dtype=complex)
    for (j, x_j) in enumerate(L):
        guess_index = tuple(x % gamma for x in y_ffts[j])
        e_index = (np.inner(x_j, b) - np.inner(y_enums[j], s_tilde_enums))
        angle = e_index* 2 * pi / q
        table[guess_index] = table[guess_index] + cos(angle) + sin(angle) * 1.j

    # Algorithm 13, Line 9: Perform the FFT on T
    fft_output = np.fft.fftn(table).real
    # Algorithm 13, Line 10: Recover the maximum of the real part of T
    return np.amax(fft_output)   

# row notation
def generate_LWE_lattice(m, n, q):
    """
    Generate an LWE instance (A, b) and secret s

    Parameters:
    - m (int): Number of samples.
    - n (int): Number of secret coefficients.
    - q (int): Modulo.

    Returns:
    - A (IntegerMatrix): LWE matrix of size n x m, where n is the number of secret coefficients and m is the number of samples
    - b (list): Target matrix of size 1 x m
    - s (list): Secret matrix of size 1 x n
    """
    B = IntegerMatrix.random(m + n, 'qary', k = m, q = q)
    A = B.submatrix(0, n, n, m + n)
    return A

def create_dual_lattice(A_lat):
    """
    Create a dual lattice from lattice A_lat

    Parameters:
    - (A_0 || A_1) (IntegerMatrix): Matrix of size m x (n - t_1)

    Returns:
    - B_dual (IntegerMatrix): Dual Lattice of size (m + n - t_1) x (m + n - t_1) and type (in row notation):
                            [I_m, (A_0 || A_1)]
                            [0, q * I_(n - t_1)]
    """
    B_dual = IntegerMatrix.identity(m + n - t_1)
    for i in range(m, m + n - t_1):
        B_dual[i, i] = B_dual[i, i] * q
    for i in range(m):
        for j in range(n - t_1):
            B_dual[i, m + j] = pow(A_lat[i, j], 1, q)

    return B_dual


def sampling(B_dual):
    """
    Implementation of Algorithm 11: Short Vectors Sampling Procedure.

    Parameters:
    - B_dual (IntegerMatrix): Dual Lattice of size (m + k_lat) x (m + k_lat).

    Returns:
    - short_vectors (List): List containing all short vectors obtained during sieving. 
    """
    global ell, D

    sieverParams = SieverParams(threads = threads, dual_mode = False)
    g6k = Siever(M = B_dual, params=sieverParams)
    short_vectors = [] # List of unique short vectors obtained in each iteration
    l_length = [] # Length of short vectors
    D = ceil(saturation_ratio * saturation_radius ** (Beta2/2.)) # The maximal number of short vectors we want to use in the attack
    for i in range(max_sieving):
        bkz(g6k, dummy_tracer, Beta1) # BKZ reduction with block size Beta1  
        short_vectors = progressive_sieve_left(g6k)
        # Check whether we have enough vectors
        if len(short_vectors) >= D:
            short_vectors = short_vectors[:D]
            short_vectors = sorted(short_vectors, key=lambda x: vector_length(x))
            l_length = [vector_length(vector) for vector in short_vectors]
            ell = sum(l_length) / len(l_length)
            required_number_of_samples(t_1, ell, sigma = 1)
            break

    print(f"Final DB size: {len(short_vectors)}")
    print(f"Average lengh l: {sum(l_length) / len(l_length)}")
    return short_vectors

def dual_attack_test(A, L, y_ffts, y_enums):
    """
    Run the distinguishing attack for one value of secret.

    Parameters:
    - A (IntegerMatrix): LWE matrix A.
    - L (list): List of short vectors sampled from the lattice B_dual.
    - y_enums (list): List of short vectors.
    - y_ffts (list): List of short vectors.

    Returns:
    - correct (list): list where 0 - correct guess, 1 incorrect guess for the given exponent
    """
    s = list()
    e = list()

    for _ in range(n):
        s.append(dist())

    for _ in range(m):    
        e.append(dist())

    b = A.multiply_left(s) # return s * A
    b = [(b[i] + e[i]) % q for i in range(m)]

    correct = [0] * sub_trials
    # Algorithm 13, Line 4: iterate of s_tilde_2 in descending order of probability:
    s_tilde_enums = dist.generate_permutations(t_1)

    for exponent in range(1, sub_trials + 1):
        power_of_two = 2 ** (exponent + 1)
        probs = []
        for i in range(len(s_tilde_enums)):
            _max = distinguisher(y_enums, y_ffts, q, p, t, L[:power_of_two], s_tilde_enums[i], b)
            probs.append((_max, s_tilde_enums[i]))

        sorted_array = sorted(probs , key=lambda x: x[0])
        index = len(sorted_array) - 1
        print("exponent: ")
        print(exponent)
        print("guess: ")
        print(sorted_array[index][1])
        print("correct: ")
        print(s[-t_1:])
        result = 1 if np.array_equal(sorted_array[index][1], s[-t_1:]) else 0
        correct[exponent - 1] = result

    return correct

def parse_parametrs():
    # Creating an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=35)
    parser.add_argument('-m', type=int, default=30)
    parser.add_argument('--t_1', type=int, default=4)
    parser.add_argument('-t', type=int, default=3)
    parser.add_argument('-q', type=int, default=3329)
    parser.add_argument('-p', type=int, default=2) # It's really the gamma from Guo and Johansson article
    parser.add_argument('--eta', type=int, default=1) # This is sigma for Gaussian distribution (when --gauss=1)

    parser.add_argument('--Beta1', type=int, default=30)
    parser.add_argument('--Beta2', type=int, default=35)
    parser.add_argument('-D', type=int, default=32)
    parser.add_argument('--gauss', type=int, default=1) # 0 - Binomial distribution, 1 - Gaussian distribution

    parser.add_argument('--saturation_radius', type=int, default=1.33)
    parser.add_argument('--saturation_ratio', type=int, default=0.95)
    parser.add_argument('--db_size_factor', type=int, default=20)
    # Parsing the command-line arguments and storing them in 'args' object
    args = parser.parse_args()
    # Assigning the parsed values to global variables for further use
    global n, m, t_1, t, q, p, eta, Beta1, Beta2, D, saturation_radius, saturation_ratio, db_size_factor, gauss

    n = args.n
    m = args.m
    t_1 = args.t_1
    t = args.t
    q = args.q
    p = args.p
    eta = args.eta
    Beta1 = args.Beta1
    Beta2 = args.Beta2
    D = args.D
    gauss = args.gauss
    saturation_radius = args.saturation_radius
    saturation_ratio = args.saturation_ratio
    db_size_factor = args.db_size_factor

if __name__ == '__main__':

    parse_parametrs()

    threads = max(1, multiprocessing.cpu_count())
    print(f"Threads: {threads}")

    # Creating folder for storing the results
    current_date = datetime.now().strftime("%m-%d-%H-%M")
    folder_path = "./guo_d_test/" + f"{n}_{m}/{gauss}/"  + f"{current_date}_{t_1}_{t}_{p}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Creating a logging file
    log_name = folder_path + "_logs.txt"
    logging.basicConfig(filename=log_name, level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    variables = {
        'gauss': gauss,
        'n': n,
        'm': m,
        'q': q,
        'p': p,
        't_1': t_1,
        't': t,
        'k_lat': k_lat,
        'Beta1': Beta1,
        'Beta2': Beta2,
        'eta': eta
    }
    
    file_name = f"variables_{n}_{m}.txt"

    # Create the file containg the LWE instance and all the variables
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        for variable, value in variables.items():
            file.write(f'{variable}:{value}\n')

    k_lat = n - t_1 - t
    if gauss == 1:
        dist = DiscreteGaussian(eta)
        logging.info(f"Gaussian distribution with sigma: {eta}")
    else:
        dist = BinomialDistribution(eta)
        logging.info(f"Binomial distribution with eta: {eta}")

    A = generate_LWE_lattice(m, n, q)
    variables['A'] = A

    # Algorithm 13, Line 1: Decompose Matrix A
    A_0_1 = A[:n - t_1]
    for i in range(n - t - t_1, n - t_1):
        for j in range(0, m):
            A_0_1[i, j] = pow(A_0_1[i, j] * p, 1, q)
    A_2 = A[-t_1:]

    # G6K uses row notation, while MATZOV utilized column notation
    A_0_1.transpose()
    A_2.transpose()

    # Algorithm 13, Line 2: Generate matrix B
    B = IntegerMatrix.identity(m + n - t_1)
    for i in range(m, m + n - t_1):
        B[i, i] = B[i, i] * q
    for i in range(0, m):
        for j in range(0, n - t_1):
            B[i, m + j] = pow(A_0_1[i, j], 1, q)

    B_dual = create_dual_lattice(A_0_1)

    # Algorithm 13, Line 3: Run the short vectors sampling algorithm
    L = sampling(B)
    sub_trials = min(int(log2(len(L))), 5) # number of sub-trials in each trial, the maximum number of vectors that will be tested is 2^sub_trials

    y_ffts = [None] * D
    y_enums = [None] * D
    multiply_left_enum = partial(multiply_left, A = A_2)

    for (i, w) in enumerate(L):
        y_ffts[i] = (w[m + n - t - t_1: m + n - t_1])

    for i in range(len(L)):
        L[i]= L[i][:m]

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, result in zip(range(len(L)), executor.map(multiply_left_enum, L)):
            y_enums[i] = result
 
    num_of_trials = 3 # how many times the test should be repreated
    logging.info(f"Number of trials: {num_of_trials}")

    s_guess = [] # An array containg the probability of sucess for each exponent    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(lambda p: dual_attack_test(*p), (A, L, y_ffts, y_enums)) for _ in range(num_of_trials)]
        for i, future in enumerate(as_completed(futures)):
            s_guess.append(future.result())

    print((f"S guess: {s_guess}"))
    logging.info(f"S guess: {s_guess}")
    s_guess = np.mean(np.array(s_guess), axis=0)
    print((f"S guess: {s_guess}"))

    plot_title = f"Effect of the number of vectors used in Dual Attack on the rank of correct s_enum\n(n={n}, m={m}, t={t}, t_1={t_1})\naverage of {num_of_trials} trials"
    plot_name = f"D_{n}_{m}"
    logging.info("Data for chart (FFT): ")
    logging.info(f"{s_guess}")
    x_values = list(range(1, len(s_guess) + 1)) 
    logging.info(f"{x_values}")
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