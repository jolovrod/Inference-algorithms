from graph_based_sampling import deterministic_eval, sub_in_vals, sub_in_vals_xy, init_XY_vals
import torch
import numpy as np
from daphne import daphne
from torch.distributions import MultivariateNormal
from copy import deepcopy
from plots import plots
import time


def U(X_vals, Y_vals, P):
    E = 0
    for v in Y_vals:
        E = E - deterministic_eval(sub_in_vals_xy(P[v][1], X_vals, Y_vals)).log_prob(Y_vals[v])
    for v in X_vals:
        E = E - deterministic_eval(sub_in_vals_xy(P[v][1], X_vals, Y_vals)).log_prob(X_vals[v])
    return E

def grad_U(X_vals, Y_vals, P):
    UX = U(X_vals, Y_vals, P)
    UX.backward()
    gradients = []
    for x in X_vals:
        gradients.append(X_vals[x].grad)
    return torch.tensor(gradients, dtype=float)

def new_Xt(Xt, eps, R):
    Xt_prime = {}
    keys_list = list(Xt.keys())
    for x in Xt:
        i = keys_list.index(x)
        Xt_prime[x] = Xt[x].detach() + eps * R[i]
        Xt_prime[x].requires_grad = True
    return Xt_prime

def leapfrog(Xt, Yt, T, eps, P, R0):
    Rt = R0 - 1/2 * eps * grad_U(Xt, Yt, P)
    for _ in range(1, T):
        Xt = new_Xt(Xt, eps, Rt)
        Rt = Rt - eps * grad_U(Xt, Yt, P)
    Xt = new_Xt(Xt, eps, Rt)
    Rt = Rt - 1/2 * eps * grad_U(Xt, Yt, P)
    return Xt, Rt

def hmc(X_vals, Y_vals, T, eps, M, P, multi_normal):

    R = multi_normal.sample()
    X_vals_prime, R_prime = leapfrog(deepcopy(X_vals), Y_vals, T, eps, P, R)
    u = np.random.uniform()
    if u < torch.exp(-H(X_vals_prime, R_prime, M, Y_vals, P) + H(X_vals, R, M, Y_vals, P)):
        X_vals = X_vals_prime
    return X_vals

def H(X, R, M, Y, P):
    # H(X, R) = U(X) + 1/2 R^T * 1/M * R
    return U(X, Y, P) + 1/2 * torch.matmul(R.T.float(), torch.matmul(M.inverse().float(), R.float()))


def hmc_sampler(graph, num_samples, T, eps):

    X0, Y0 = init_XY_vals(graph, requires_grad=True)

    M = torch.eye(len(X0))

    P = graph[1]['P']

    multi_normal = MultivariateNormal(torch.zeros(len(M)), M)
    X_samples = []
    return_values = []
    prob_sequence = []

    X_vals = X0

    for _ in range(num_samples):
        X_vals = hmc(X_vals, Y0, T, eps, M, P, multi_normal)
        X_samples.append(X_vals)
        return_values.append(deterministic_eval(sub_in_vals(graph[-1], X_vals)))
        prob_sequence.append(-U(X_vals, Y0, P))

    return return_values, prob_sequence


if __name__ == '__main__':

    num_samples = 20000
    T = 10
    eps = .1

    for i in [1,2,5]:
        graph = daphne(['graph','-i','C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/{}.daphne'.format(i)])

        print("\n\n\nprogram", i)
        start_time = time.time()
        samples, prob_sequence  = hmc_sampler(graph, num_samples, T, eps)
        print("Time for program", i, ":   ", round(time.time()-start_time, 2), "seconds")

        plots(i, samples, prob_sequence, "_hmc")


    