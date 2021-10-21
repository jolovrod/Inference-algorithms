import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev, mean

from numpy.core.fromnumeric import std

def sampler(i, samples, st_or_gr):
    num_samples = len(samples)
    print("\n\n\nprogram,", i)

    if i == 1:
        for _ in range(num_samples):
            samples = [float(x) for x in samples]
        
        plt.figure(figsize=(2,1.5))
        plt.xlabel("mu")
        plt.ylabel("frequency")
        plt.hist(samples)
        figstr = "histograms/program_"+str(i) + st_or_gr
        plt.savefig(figstr)

        print("mu:\n{:.5f} \n{:.5f} \n".format(mean(samples), stdev(samples)))


    elif i==2:
        for n in range(num_samples):
            samples[n] = [float(x) for x in samples[n]]
        variables = np.array(samples,dtype=object).T.tolist()
        for d in range(len(variables)):
            plt.figure(figsize=(5,4))
            plt.hist(variables[d])
            if d==0:
                plt.xlabel("slope")
            else:
                plt.xlabel("bias")
            plt.ylabel("frequency")
            figstr = "histograms/program_"+str(i)+"_var_"+str(d)+st_or_gr
            plt.savefig(figstr)

        means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
        stds = ["{:.5f}".format(stdev(variables[d])) for d in range(len(variables))]
        print("mean", means)
        print("stds", stds)

    elif i==3:
        for n in range(num_samples):
            samples[n] = [int(x) for x in samples[n]]
        variables = np.array(samples,dtype=object).T.tolist()
        for d in range(len(variables)):
            counts = [0,0,0]
            for element in variables[d]:
                counts[element] += 1
            plt.figure(figsize=(5,4))
            plt.bar([0,1,2],counts)
            plt.xlabel("states["+str(d)+"]")
            plt.ylabel("frequency")
            figstr = "histograms/program_"+str(i)+"_var_"+str(d)+st_or_gr
            plt.savefig(figstr)
        means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
        stds = ["{:.5f}".format(stdev(variables[d])) for d in range(len(variables))]
        print("mean", means)
        print("stds", stds)



    else:
        W0, b0, W1, b1 = [],[],[],[]
        for n in range(num_samples):
            W0_n, b0_n, W1_n, b1_n = samples[n]
            W0.append(W0_n.numpy().flatten())
            b0.append(b0_n.numpy().flatten())
            W1.append(W1_n.numpy().flatten())
            b1.append(b1_n.numpy().flatten())
        
        objects = [W0, b0, W1, b1]
        strs = ["W0", "b0", "W1", "b1"]

        for j in range(4):
            print("\n\n",strs[j])
            variables = np.array(objects[j]).T.tolist()
            for d in range(len(variables)):
                plt.figure(figsize=(5,4))
                plt.hist(variables[d])
                plt.xlabel(strs[j]+"[{:d}]".format(d))
                plt.ylabel("frequency")
                figstr = "histograms/program_"+str(i)+"_"+strs[j]+"_"+str(d)+st_or_gr
                plt.savefig(figstr)

            means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
            stds = ["{:.5f}".format(stdev(variables[d])) for d in range(len(variables))]
            print("mean", means)
            print("stds", stds)