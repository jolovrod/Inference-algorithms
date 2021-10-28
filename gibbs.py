from graph_based_sampling import deterministic_eval, sub_in_vals, sub_in_vals_xy, init_X_vals
import torch
import numpy as np
from daphne import daphne
import time
from plots import plots


def accept(x, X_vals_prime, X_vals, P, inv_free, Y):
    # https://www.cs.ubc.ca/~fwood/CS532W-539W/lectures/11.pdf
    # Algorithm 1 : lines 3-11
    # X_vals, X_vals' should only differ at the coordinate of x. 
    exp = P[x][1]

    log_alpha = deterministic_eval(sub_in_vals(exp, X_vals_prime)).log_prob(X_vals[x]) \
        - deterministic_eval(sub_in_vals(exp, X_vals)).log_prob(X_vals_prime[x])
    V_x = inv_free[x]
    
    for v in V_x:
        exp = P[v][1]
        if v in Y:
            y_observed = torch.tensor(Y[v])
            log_alpha += deterministic_eval(sub_in_vals_xy(exp, X_vals_prime, Y)).log_prob(y_observed)
            log_alpha -= deterministic_eval(sub_in_vals_xy(exp, X_vals, Y)).log_prob(y_observed)
        else:
            log_alpha += deterministic_eval(sub_in_vals(exp, X_vals_prime)).log_prob(X_vals_prime[v])
            log_alpha -= deterministic_eval(sub_in_vals(exp, X_vals)).log_prob(X_vals[v])
        
    return np.exp(log_alpha)

def log_probability(X_vals, Y, P):
    logalph = 0
    for v in Y:
        y_observed = torch.tensor(Y[v])
        logalph = logalph + deterministic_eval(sub_in_vals_xy(P[v][1], X_vals, Y)).log_prob(y_observed)
    for v in X_vals:
        logalph = logalph + deterministic_eval(sub_in_vals_xy(P[v][1], X_vals, Y)).log_prob(X_vals[v])
    return logalph

def gibbs_step(X, P, X_vals, inv_free, Y):
    for x in X:
        exp = P[x][1]
        d = deterministic_eval(sub_in_vals(exp, X_vals))
        X_vals_prime = X_vals.copy()
        X_vals_prime[x] = d.sample()
        alpha = accept(x, X_vals_prime, X_vals, P, inv_free, Y)
        u = np.random.uniform()
        if u < alpha:
            X_vals = X_vals_prime
        prob_val = log_probability(X_vals, Y, P)
    return X_vals, prob_val

def gibbs(P, X, X_vals, inv_free, n_steps, Y):
    sequence = []
    prob_sequence = []
    for i in range(n_steps):
        if i%1000==0:
            print("step", i)
        X_vals, prob_val = gibbs_step(X, P, X_vals, inv_free, Y)
        sequence.append(X_vals)
        prob_sequence.append(prob_val)
    return sequence, prob_sequence

def gibbs_sampler(graph, n_steps):
    # graph [{}, VAPY, observes]
    # V : vertices, A : arcs, P : link functions, Y : observations
    V, A, P, Y = graph[1]['V'], graph[1]['A'], graph[1]['P'], graph[1]['Y']
    X = list(set(V) - set(Y))
    inv_free = inv_free_vars(X,A) 
    samples = []
    X_vals = init_X_vals(graph)
    values, prob_sequence = gibbs(P, X, X_vals, inv_free, n_steps, Y)
    for v in values:
        # The last entry in the graph is the return value of the program.
        samples.append(deterministic_eval(sub_in_vals(graph[-1], v)))
    return samples, prob_sequence

def inv_free_vars(X, A):    # inverse free variables function
    inv_free = {x:[x] for x in X}
    for v in X:  
        if v in A:
            inv_free[v] += A[v]
    return inv_free

if __name__ == '__main__':

    num_samples = 20000

    for i in range(1,6):
        graph = daphne(['graph','-i','C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/{}.daphne'.format(i)])

        print("\n\n\nprogram", i)
        start_time = time.time()
        samples, prob_sequence  = gibbs_sampler(graph, num_samples)
        print("Time for program", i, ":   ", round(time.time()-start_time, 2), "seconds")

        plots(i, samples, prob_sequence, "_MH_gibbs")
        print("\n\n\n")

    