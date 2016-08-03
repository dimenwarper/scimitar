import numpy as np

def cohens_d(x, y):
            return (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))
