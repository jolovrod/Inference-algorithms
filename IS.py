from evaluation_based_sampling import evaluate_program
from daphne import daphne
from plots import plots
import time


def likelihood_weighting(ast, L):
    samples = []
    logW = []
    for _ in range(L):
        sigma = {'logW': 0}
        samples_l, sigma = evaluate_program(ast, sigma)
        logW_l = sigma['logW']
        samples.append(samples_l)
        logW.append(logW_l)   
    return samples, logW


def IS(ast, L):
    samples, logW = likelihood_weighting(ast,L)
    return samples, logW


if __name__ == '__main__':

    num_samples = 20000

    for i in range(1,6):
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/{}.daphne'.format(i)])

        print("\n\n\nprogram", i)
        start_time = time.time()
        samples, logW = IS(ast, num_samples)
        plots(i, samples, logW, "_IS")
        print("Time for program", i, ":   ", round(time.time()-start_time, 2), "seconds")
        print("\n\n\n")



