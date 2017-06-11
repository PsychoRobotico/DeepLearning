import numpy as np


J = 1  # coupling strength
L = 32  # grid size

temperatures = np.arange(1, 3.51, 0.1)
ntherm = 100  # thermalization steps
nsamples = 100  # number of samples for every starting configuration
nbin = 10  # number of starting configurations

n = len(temperatures) * nbin * nsamples  # total number of samples

C = np.empty((n, L, L))
T = np.empty(n)

# checkerboard pattern
x = np.zeros((L, L), dtype=int)
x[1::2, 0::2] = 1
x[0::2, 1::2] = 1

c = 0
for t in temperatures:
    print(t)
    for i in range(nbin):
        # start with polarized state
        grid = np.ones((L, L)) * (-1)**(i % 2)
        # draw a number of sample configurations
        for j in range(nsamples):
            # perform a number of updates until thermalization
            for k in range(ntherm):
                neighbors = np.roll(grid, +1, axis=0) + \
                            np.roll(grid, -1, axis=0) + \
                            np.roll(grid, +1, axis=1) + \
                            np.roll(grid, -1, axis=1)
                # calculate the potential changes in energy
                dE = 2 * J * (grid * neighbors)
                # calculate the transition probabilities
                p = np.exp(-dE / t)
                # decide which transitions will occur
                # (avoid updating neighbors using alternating checkerboard pattern)
                grid *= 1 - 2 * np.multiply(np.int8(np.random.rand(L, L) < p), x ^ (k % 2))

            C[c] = grid
            T[c] = t
            c += 1

# shuffle and save data
p = np.random.permutation(len(C))
np.savez_compressed('data.npz', X=C[p], y=T[p])
