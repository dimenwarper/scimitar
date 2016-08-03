from pyroconductor.glasso import glasso
import numpy as np

r = np.random.randn(3, 10)
s= np.cov(r.T)

print glasso(s, 0.1)

