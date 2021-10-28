import matplotlib.pyplot as plt
import numpy as np
from statistics import variance, mean


def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum()

def plots(i, samples, prob_sequence, alg):
    num_samples = len(samples)

    if i in [3,4]:
        
        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)

        else:
            means = mean(np.array(samples, dtype=float))
            vars = variance(np.array(samples, dtype=float))

        print("mean", means)
        print("variance", vars)


    if i == 1:
        for _ in range(num_samples):
            samples = [float(x) for x in samples]
        
        plt.figure(figsize=(5,4))
        plt.xlabel("mu")
        plt.ylabel("frequency")
        plt.title("Histogram program 1" + alg)
        plt.hist(samples)
        figstr = "histograms/program_"+str(i) + alg
        plt.savefig(figstr)


        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("mu")
        plt.title("Trace plot program 1" + alg)
        plt.plot(samples)
        figstr = "trace_plots/program_"+str(i) + alg
        plt.savefig(figstr)


        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("log prob")
        plt.title("log prob plot program 1 " + alg)
        plt.plot(prob_sequence)
        figstr = "prob_plots/program_"+str(i) + alg
        plt.savefig(figstr)

        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)
        else:
            means = mean(np.array(samples, dtype=float))
            vars = variance(np.array(samples, dtype=float))

        print("mean", means)
        print("variance", vars)


    elif i in [2,5]:
        for n in range(num_samples):
            samples[n] = [float(x) for x in samples[n]]
        
        variables = np.array(samples,dtype=object).T.tolist()
        for d in range(len(variables)):
            plt.figure(figsize=(5,4))
            plt.hist(variables[d])
            if d==0 and i==2:
                xvarname = "slope"
            elif d==0 and i==5:
                xvarname = "x"
            elif d==1 and i==2:
                xvarname = "bias"
            else:
                xvarname = "y"

            plt.xlabel(xvarname)
            plt.ylabel("frequency")
            figstr = "histograms/program_"+str(i)+"_var_"+str(d)+alg
            plt.savefig(figstr)

            plt.figure(figsize=(5,4))
            plt.xlabel("Iterations")
            plt.ylabel(xvarname)
            plt.title("Trace plot program " + str(i) + " " + xvarname + " " + alg)
            plt.plot(variables[d])
            figstr = "trace_plots/program_"+str(i)+"_var_"+str(d)+alg
            plt.savefig(figstr)


        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("log prob")
        plt.title("log prob plot program " + str(i) + alg)
        plt.plot(prob_sequence)
        figstr = "prob_plots/program_"+str(i)+alg
        plt.savefig(figstr)

        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)
        else:
            means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
            vars = ["{:.5f}".format(variance(variables[d])) for d in range(len(variables))]

        print("mean", means)
        print("variance", vars)