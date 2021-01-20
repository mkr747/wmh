import numpy as np

class Test2:
    def __init__(self):
        self.data_names = ['AMZN 1Y.csv', 'AMZN 3M.csv', 'AMZ 6M.csv', 'apple 1Y', 'apple 3M.csv', 'apple 6M.csv', 'FB 1Y.csv', 'FB 3M.csv', 'FB 6M.csv']
        self.cvs = np.arrange(1, 10)
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        self.degrees = np.arrange(1, 10, 0.1)
        self.gammas = np.arrange(0.0, 1.0, 0.01)
        self.coef0s = np.arrange(0, 20, 0.1)
        self.tols = np.arrage(0, 5, 0.001)
        self.cs = np.arrange(0.0, 100)
        self.epsilon = np.arrange(0.0, 5, 0.01)
        self.shrinking = [True, False]
        self.verbose = [False]

    def test(self):
        for data_name in self.data_names:
            for cv in self.cvs:
                for kernel in self.kernels:
                    for degree in self.degrees:
                        for gamma in self.gammas:
                            for coef0 in self.coef0s:
                                for tol in self.tols:
                                    for c in self.cs:
                                        for epsilon in self.epsilon:
                                            for shrinking in self.shrinking:
                                                pass
