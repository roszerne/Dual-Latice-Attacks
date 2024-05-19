import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os
import multiprocessing
import resource
import argparse
import math
from scipy.stats import binom, norm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt, ceil, pi, sin, cos, exp, log, log2, comb, prod
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
from g6k.utils.stats import dummy_tracer
from fpylll import IntegerMatrix, BKZ
from fpylll.util import gaussian_heuristic
from fpylll.algorithms.bkz2 import BKZReduction
from datetime import datetime
from distributions import BinomialDistribution, DiscreteGaussian
from utils import vector_length, print_memory_usage, multiply_left
# Matzov - column notation, g6k - row notation 

n = None
m = None
q = None
p = None
k_enum = None
k_fft = None
k_lat = None
# attack parameters
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
l_lengths = [] # List of size num_of_trials x D, containing lengths of vectors for each trial
l_exp = None# List of size num_of_trials, containing average size of D vectors for each trial 
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

def D_of_advantage(l_exp):
    D_eq_tilde = exp(4 * pow(((pi * dist.sigma * l_exp) / q), 2))
    phi_fp = dist.CDF_INV((1 - mi / (2 * pow(2, k_enum * dist.entropy) * pow(p, k_fft))))
    phi_fn = dist.CDF_INV(1 - mi / 2)
    D_fpfn_tilde = pow((phi_fp + phi_fn), 2)
    D_round_tilde = prod((sin(pi * s / p) / (pi * s / p)) ** (-2 * k_fft * dist.probability_distribution[s]) for s in dist.probability_distribution.keys() if pow(s, 1, p) != 0)
    D_est = D_eq_tilde + D_round_tilde + D_fpfn_tilde + (1 / 2)
    print(f"Minimum D required: {D_est}")
    logging.info(f"Minimum D required: {D_est}")
    return ceil(D_est)

def C_of_advantage(D):
    phi_fp = dist.CDF_INV((1 - (mi / (2 * pow(2, k_enum * dist.entropy) * pow(p, k_fft)))))
    print(f"Based on: {D}")
    C_estimate = phi_fp * sqrt( (1/2) * D)
    print(f"C for {mi} advantage: {C_estimate}")
    logging.info(f"C for {mi} advantage: {C_estimate}")
    return C_estimate

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
    
    '''print("hk3 sieving with size")
    logging.info("hk3 sieving with size")
    with g6k.temp_params(saturation_ratio = saturation_ratio, saturation_radius = saturation_radius, db_size_base=sqrt(saturation_radius) ,db_size_factor = db_size_factor):
        g6k(alg="hk3")
    print("Finished hk3")'''
    g6k.resize_db(ceil(saturation_ratio * saturation_radius ** (Beta2/2.)))
    print("Finished resizing")
    while g6k.l > 0:
        print_memory_usage()
        # Extend the lift context to the left
        g6k.extend_left(1)
        print(f"Extending: {g6k.l}")
    print("Finished extending")
    db = list(g6k.itervalues())
    print("Finished listing")
    database = [None] * len(db)
    multiply_left_partial = partial(multiply_left, A = g6k.M.B[g6k.l:g6k.r])
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Wykonanie funkcji multiply_left(v) na każdym elemencie z L i wypełnienie tablicy y_fft
        for i, result in zip(range(len(db)), executor.map(multiply_left_partial, db)):
            database[i] = result

    print(f"Final database size: {len(database)}")
    logging.info(f"Final database size:: {len(database)}")

    return [w[:m] for w in database] + [[-x for x in w[:m]] for w in database]

def mod_switch_distinguisher(y_enums, y_ffts, q, p, k_fft,  L, s_tilde_enum, b):

    table = np.zeros(shape=(p,) * k_fft, dtype=complex)

    for (j, x_j) in enumerate(L):
        index = tuple(int(round(1. * p * x / q) % p) for x in y_ffts[j])
        inner_product = np.inner(x_j, b) - np.inner(y_enums[j], s_tilde_enum)
        angle = inner_product * 2 * pi / q
        table[index] += cos(angle) + sin(angle) * 1.j

    fft_output = np.fft.fftn(table).real
    return np.any(fft_output > C), np.amax(fft_output) 

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
    global l_lengths, l_exp, D

    sieverParams = SieverParams(threads = threads, dual_mode = False)
    g6k = Siever(M = B_dual, params=sieverParams)
    short_vectors = [] # List of unique short vectors obtained in each iteration
    l_length = [] # Length of short Vectors
    D = ceil(saturation_ratio * saturation_radius ** (Beta2/2.))
    for i in range(max_sieving):
        logging.info(f"Begging BKZ reduction")
        bkz(g6k, dummy_tracer, Beta1) # BKZ reduction with block size Beta1  
        logging.info(f"BKZ reduction completed")
        new_short_vectors = progressive_sieve_left(g6k)
        # We only want unique short vectors
        for vector in new_short_vectors:
            if vector not in short_vectors:
                short_vectors.append(vector)
        logging.info(f"New vectors sampled, number of vectors: {len(short_vectors)}")
        # Check whether we have enough vectors
        if len(short_vectors) >= D:
            short_vectors = short_vectors[:D]
            short_vectors = sorted(short_vectors, key=lambda x: vector_length(x))
            l_length = [vector_length(vector) for vector in short_vectors]
            l_exp = max(l_length)
            D_of_advantage(l_exp)
            break

    l_lengths.append(l_length)
    print("Final DB size: ")
    print(len(short_vectors))
    logging.info(f"Vectors used\n: {short_vectors}")
    logging.info(f"Average lengh l: {sum(l_length) / len(l_length)}")
    return short_vectors

def dual_attack_test(A, L, y_ffts, y_enums):
    s = list()
    e = list()

    for _ in range(n):
        s.append(dist())

    for _ in range(m):    
        e.append(dist())

    b = A.multiply_left(s) # return s * A
    b = [(b[i] + e[i]) % q for i in range(m)]
    # Line 4:
    s_tilde_enums = dist.generate_permutations(k_enum)
    probs = []
    for i in range(len(s_tilde_enums)):
        T, fmax = mod_switch_distinguisher(y_enums, y_ffts, q, p, k_fft, L, s_tilde_enums[i], b)
        probs.append((fmax, s_tilde_enums[i]))
        if T:
            print("guess: ")
            print(s_tilde_enums[i])
            print("correct: ")
            print(s[:k_enum])
            #print(probs)
            result = 1 if np.array_equal(s_tilde_enums[i], s[:k_enum]) else 0
            print("result: ")
            print(result)
            return result
    print("No result")
    return 0

def parse_parametrs():
    # Creating an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=30)
    parser.add_argument('-m', type=int, default=25)
    parser.add_argument('--kenum', type=int, default=4)
    parser.add_argument('--kfft', type=int, default=4)
    parser.add_argument('-q', type=int, default=3329)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('--eta', type=int, default=1) # sigma for Gaussian distribution

    parser.add_argument('--Beta1', type=int, default=25)
    parser.add_argument('--Beta2', type=int, default=30)
    parser.add_argument('-D', type=int, default=64)
    parser.add_argument('-gauss', type=int, default=1) # 0 - Binomial distribution, 1 - Gaussian distribution

    parser.add_argument('--saturation_radius', type=int, default=1.33)
    parser.add_argument('--saturation_ratio', type=int, default=0.95)
    parser.add_argument('--db_size_factor', type=int, default=20)
    # Parsing the command-line arguments and storing them in 'args' object
    args = parser.parse_args()
    # Assigning the parsed values to global variables for further use
    global n, m, k_enum, k_fft, q, p, eta, Beta1, Beta2, D, saturation_radius, saturation_ratio, db_size_factor, gauss

    n = args.n
    m = args.m
    k_enum = args.kenum
    k_fft = args.kfft
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
    folder_path = "./c_prob_test/" + f"{current_date}_{n}_{m}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Creating a logging file
    log_name = folder_path + "_logs.txt"
    logging.basicConfig(filename=log_name, level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')

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
    
    file_name = f"variables_{n}_{m}.txt"

    # Create the file containg the LWE instance and all the variables
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        for variable, value in variables.items():
            file.write(f'{variable}:{value}\n')

    k_lat = n - k_enum - k_fft
    if gauss == 1:
        dist = DiscreteGaussian(eta)
        logging.info(f"Gaussian distribution with sigma: {eta}")
    else:
        dist = BinomialDistribution(eta)
        logging.info(f"Binomial distribution with eta: {eta}")

    A = generate_LWE_lattice(m, n, q)
    variables['A'] = A

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
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of sieving:", execution_time, "seconds")
    logging.info(f"Sampling completed, execution time: {execution_time}")
    #C = parameters_of_advantage(l_exp)
    #required_number_of_samples(s, e, l_exp)
    d = len(L)
    y_ffts = [None] * d
    y_enums = [None] * d
    multiply_left_fft = partial(multiply_left, A = A_fft)
    multiply_left_enum = partial(multiply_left, A = A_enum)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Wykonanie funkcji multiply_left(v) na każdym elemencie z L i wypełnienie tablicy y_fft
        for i, result in zip(range(len(L)), executor.map(multiply_left_fft, L)):
            y_ffts[i] = result
        for i, result in zip(range(len(L)), executor.map(multiply_left_enum, L)):
            y_enums[i] = result

    D_of_advantage(l_exp)
    C = C_of_advantage(D)

    num_of_trials = 100 # how many times the test should be repreated
    '''
    s_guess = [None] * num_of_trials # An array containg the ranks for each trial    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(lambda p: dual_attack_test(*p), (A, L, y_ffts, y_enums)) for _ in range(num_of_trials)]
        for i, future in enumerate(as_completed(futures)):
            s_guess[i] = future.result()'''


    logging.info(f"Number of trials: {num_of_trials}")
    s_guess = []
    for i in range (num_of_trials):
        s_guess.append(dual_attack_test(A, L, y_ffts, y_enums))

    logging.info(f"S_guess: {s_guess}")
    logging.info(f"S_guess %: {sum(s_guess) / len(s_guess)}")
    print(f"S_guess: {s_guess}")
    print(f"S_guess %: {sum(s_guess) / len(s_guess)}")