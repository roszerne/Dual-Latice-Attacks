"""
This script contains the code for the experiments discussed in Chapter 5, Section 3.
It examines the bound on the cutoff value C for the dual attack proposed by MATZOV in Theorem 4.3.1.
"""
import numpy as np
import logging
import os
import multiprocessing
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt, ceil, pi, sin, cos, exp, prod
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
from g6k.utils.stats import dummy_tracer
from fpylll import IntegerMatrix
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

def required_C_and_D(secret, error, ell):
    """
    Calculate the required number of short vectors and the approperiate value for C according to Theorem 4.3.1

    Parameters:
    - ell (float): Average lenght of short vector sampled from the lattice.
    - secret (list): secret vector.
    - error (list): error vector.

    Returns: D and C
    """
    # assuming alpha = 1, since both error and secret are from the same distribution
    tau = ((np.inner(error, error)+ np.inner(secret[-k_lat:], secret[-k_lat:])) / (m + k_lat)) * pow(ell, 2)
    D_eq = exp(4 * pow((pi/ q), 2) * tau)
    print(f"D eq: {D_eq}")
    D_round = pow(prod((sin(pi * s / p) / (pi * s / p)) for s in secret[k_enum:k_enum + k_fft] if pow(s, 1, p) != 0), -2)
    print(f"D round: {D_round}")
    D_arg = 0.5 + exp(-8 * pow((pi / q), 2) * tau)
    print(f"D arg: {D_arg}")
    phi_fp = dist.CDF_INV((1 - mi / (2 * dist.n_enum(secret[:k_enum]) * pow(p, k_fft))))
    print(f"phi_fp: {phi_fp}")
    phi_fn = dist.CDF_INV(1 - mi / 2)
    print(f"phi_fn: {phi_fn}")
    D_fpfn = pow((phi_fp + phi_fn), 2)
    print(f"D fpfn: {D_fpfn}")
    D_estimate = D_eq + D_round + D_arg + D_fpfn 
    phi_fp = dist.CDF_INV((1 - mi / (2 * dist.n_enum(secret[:k_enum]) * pow(p, k_fft))))
    C_estimate = phi_fp * sqrt(D_arg + D)
    print(f"C (Concrete): {C_estimate}")
    print(f"D (Concrete): {D_estimate}")
    return D_estimate, C_estimate

def D_of_advantage(ell):
    """
    Calculate the required number of short vectors according to Theorem 4.3.2

    Parameters:
    - ell (float): Average lenght of short vector sampled from the lattice.

    Returns: D
    """
    D_eq_tilde = exp(4 * pow(((pi * dist.sigma * ell) / q), 2))
    phi_fp = dist.CDF_INV((1 - mi / (2 * pow(2, k_enum * dist.entropy) * pow(p, k_fft))))
    phi_fn = dist.CDF_INV(1 - mi / 2)
    D_fpfn_tilde = pow((phi_fp + phi_fn), 2)
    D_round_tilde = prod((sin(pi * s / p) / (pi * s / p)) ** (-2 * k_fft * dist.probability_distribution[s]) for s in dist.probability_distribution.keys() if pow(s, 1, p) != 0)
    D_est = np.real(D_eq_tilde) + np.real(D_round_tilde) + np.real(D_fpfn_tilde) + (1 / 2)
    return ceil(D_est)

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
    logging.info("Finished hk3")
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

    return [w[:m] for w in database]

def distinguisher(y_enums, y_ffts, q, p, k_fft, L, s_tilde_enum, b):
    """
    Implementation of the distinguisher of MATZOV

    Parameters:
    y_enums (list): List of short vectors
    y_ffts (list): List of short vectors
    q (int): Modulus of LWE instance
    p (int): Modulus of FFT distinguisher
    k_fft (int): number of secret coordinates we enumerate over using FFT
    L (list): List of short vectors from the lattice
    s_tilde_enums (list): current guess on s_enums
    b (list): b vector from an LEW instance

    Returns:
    np.any(fft_output > C) (Boolean): The maximum of the real part of Ts.

    """
    table = np.zeros(shape=(p,) * k_fft, dtype=complex) # Algorithm 10, Line 5: Initialize a table T

    # Algorithm 10, Line 6: for every short vector (x, y):
    for (j, x_j) in enumerate(L):
        guess_index = tuple(int(round(p * v * 1.0 / q) % p) for v in y_ffts[j])
        e_index = np.inner(x_j, b) - np.inner(y_enums[j], s_tilde_enum)
        e_index = e_index * 2 * pi / q
        table[guess_index] = table[guess_index] + cos(e_index) + sin(e_index) * 1.j

    fft_output = np.fft.fftn(table).real
    return np.any(fft_output > C), np.amax(fft_output)   

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
    Algorithm 10, Line 2: Create a dual lattice from lattice A_lat

    Parameters:
    - A_lat (IntegerMatrix): Matrix of size m x k_lat

    Returns:
    - B (IntegerMatrix): Dual Lattice of size (m + k_lat) x (m + k_lat) and type (in row notation):
                            [I_m, A_lat]
                            [0, q * I_kat]
    """
    B = IntegerMatrix.identity(m + k_lat)
    for i in range(m, m + k_lat):
        B[i, i] = B[i, i] * q
    for i in range(m):
        for j in range(k_lat):
            B[i, m + j] = pow(A_lat[i, j], 1, q)

    return B

def sampling(B_dual):
    """
    Implementation of Algorithm 11: Short Vectors Sampling Procedure.

    Parameters:
    - B_dual (IntegerMatrix): B_dual (IntegerMatrix): Dual Lattice of size (m + k_lat) x (m + k_lat).

    Returns:
    - short_vectors (List): List containing all short vectors obtained during sieving. 
    """
    global ell, D

    sieverParams = SieverParams(threads = threads, dual_mode = False)
    g6k = Siever(M = B_dual, params=sieverParams)
    short_vectors = [] # List of unique short vectors obtained in each iteration
    l_length = [] # Length of short Vectors
    D = ceil(saturation_ratio * saturation_radius ** (Beta2/2.))
    for i in range(max_sieving):
        bkz(g6k, dummy_tracer, Beta1) # BKZ reduction with block size Beta1  
        short_vectors = progressive_sieve_left(g6k)
        # Check whether we have enough vectors
        if len(short_vectors) >= D:           
            short_vectors = sorted(short_vectors, key=lambda x: vector_length(x))
            short_vectors = short_vectors[:D]
            l_length = [vector_length(vector) for vector in short_vectors]
            ell = sum(l_length) / len(l_length)
            D_of_advantage(ell)
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
    - (is_correct_larger, result) (tuple): l
    is_correct_larger: 1 if the FFT for correct guess on s_enum is larger that C, 0 otherwise
    result: 0 - correct guess, 1 incorrect guess
    """
    global C
    s = list()
    e = list()

    for _ in range(n):
        s.append(dist())

    for _ in range(m):    
        e.append(dist())

    b = A.multiply_left(s) # return s * A
    b = [(b[i] + e[i]) % q for i in range(m)]

    _, C = required_C_and_D(s, e, ell)

    # First check, whether F(s_enum, s_fft) (correct score) is > C
    is_correct_larger = 0
    _, f_max_correct = distinguisher(y_enums, y_ffts, q, p, k_fft, L, np.array(s[:k_enum]), b)
    if f_max_correct > C:
        is_correct_larger = 1
    # Algorithm 10, Line 4: for every value of  s_tilde_enum, in descending order of probability according to secret distribution:
    s_tilde_enums = dist.generate_permutations(k_enum)
    probs = []
    for i in range(len(s_tilde_enums)):
        T, fmax = distinguisher(y_enums, y_ffts, q, p, k_fft, L, s_tilde_enums[i], b)
        probs.append((fmax, s_tilde_enums[i]))
        if T:
            print(f"guess: {s_tilde_enums[i]}")
            logging.info(f"guess: {s_tilde_enums[i]}")
            print(f"correct: {s[:k_enum]}")
            logging.info(f"correct: {s[:k_enum]}")
            result = 1 if np.array_equal(s_tilde_enums[i], s[:k_enum]) else 0
            print(f"result: {result}")
            logging.info(f"result: {result}")
            return (is_correct_larger, result)
        elif np.array_equal(s_tilde_enums[i], s[:k_enum]):
            print("Passed the correct value")
            logging.info("Passed the correct value")
            return (is_correct_larger, 0)
    print("No result")
    logging.info("No result")
    return (is_correct_larger, 0)

def parse_parametrs():
    # Creating an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=35)
    parser.add_argument('-m', type=int, default=30)
    parser.add_argument('--kenum', type=int, default=5)
    parser.add_argument('--kfft', type=int, default=5)
    parser.add_argument('-q', type=int, default=3329)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('--eta', type=int, default=1) # This is sigma for Gaussian distribution (when --gauss=1)

    parser.add_argument('--Beta1', type=int, default=20)
    parser.add_argument('--Beta2', type=int, default=35)
    parser.add_argument('-D', type=int, default=64)
    parser.add_argument('--gauss', type=int, default=1) # 0 - Binomial distribution, 1 - Gaussian distribution

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
    folder_path = "./c_test/" + f"{n}_{m}/" + f"{current_date}_{k_enum}_{k_fft}_{p}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Creating a logging file
    log_name = folder_path + "_logs.txt"
    logging.basicConfig(filename=log_name, level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    k_lat = n - k_enum - k_fft
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

    if gauss == 1:
        dist = DiscreteGaussian(eta)
        logging.info(f"Gaussian distribution with sigma: {eta}")
    else:
        dist = BinomialDistribution(eta)
        logging.info(f"Binomial distribution with eta: {eta}")

    A = generate_LWE_lattice(m, n, q)
    variables['A'] = A


    # Algorithm 10, Line 1: compute A_lat composed of last k_enum columns  
    A_enum = A[:k_enum]
    A_fft = A[k_enum: k_enum + k_fft]
    A_lat = A[k_enum + k_fft:]

    # G6K uses row notation, while MATZOV utilized column notation
    A_enum.transpose()
    A_fft.transpose()
    A_lat.transpose()

    # Algorithm 10, Line 2 : Compute the matrix B 
    B_dual = create_dual_lattice(A_lat)
    
    # Algorithm 10, Line 3 : Run the short vectors sampling algorithm on the basis B
    L = sampling(B_dual)

    d = len(L)
    y_ffts = [None] * d
    y_enums = [None] * d
    multiply_left_fft = partial(multiply_left, A = A_fft)
    multiply_left_enum = partial(multiply_left, A = A_enum)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, result in zip(range(len(L)), executor.map(multiply_left_fft, L)):
            y_ffts[i] = result
        for i, result in zip(range(len(L)), executor.map(multiply_left_enum, L)):
            y_enums[i] = result

    num_of_trials = 3 # how many times the test should be repreated

    s_guess = [None] * num_of_trials # An array containg the ranks for each trial    
    are_correct_larger = [None] * num_of_trials
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(lambda p: dual_attack_test(*p), (A, L, y_ffts, y_enums)) for _ in range(num_of_trials)]
        for i, future in enumerate(as_completed(futures)):
            are_correct_larger[i] = future.result()[0]
            s_guess[i] = future.result()[1]
            
    logging.info(f"Number of trials: {num_of_trials}")
    logging.info(f"S_guess: {s_guess}")
    logging.info(f"S_guess len: {len(s_guess)}")
    logging.info(f"S_guess %: {sum(s_guess) / len(s_guess)}")
    logging.info(f"Are_correct_larger: {are_correct_larger}")
    logging.info(f"Are_correct_larger %: {sum(are_correct_larger) / len(are_correct_larger)}")
    print(f"S_guess: {s_guess}")
    print(f"S_guess %: {sum(s_guess) / len(s_guess)}")
    print(f"Are_correct_larger: {are_correct_larger}")
    print(f"Are_correct_larger %: {sum(are_correct_larger) / len(are_correct_larger)}")

