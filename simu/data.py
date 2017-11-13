import numpy as np

def simu_data(N, D, K, sigma=1):
  Y = np.zeros((D, N))
  U = np.random.normal(0.0, 2.0, size=(D, K))
  V = np.random.normal(0.0, 1.0, size=(K, N))
  mean = np.dot(U, V)

  for d in range(D):
    for n in range(N):
      Y[d, n] = np.random.normal(mean[d, n], sigma)

  return Y


if __name__ = "__main__":
	Y = simu_data(N = 100, D = 15, K = 5, sigma = 1)
