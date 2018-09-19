import numpy as np


def gen_kernel(m):
    p1, p2, lam, a, b, c, d = m

    q = np.random.multinomial(1, [p1, p2, 1 - p1 - p2])
    x1 = q[0] * np.random.poisson(lam)
    x2 = q[1] * np.random.normal(a, b)
    x3 = q[2] * np.random.normal(c, d)
    x = x1 + x2 + x3
    if x < 0:
        x = -x
    return x + 10000


def gen_kernel_n(m, n):
    p1, p2, lam, a, b, c, d = m

    q = np.random.multinomial(1, [p1, p2, 1 - p1 - p2], n)
    x1 = q[:, 0] * np.random.poisson(lam, n)
    x2 = q[:, 1] * np.random.normal(a, b, n)
    x3 = q[:, 2] * np.random.normal(c, d, n)
    x = x1 + x2 + x3
    reverse = x < 0
    x[reverse] = -x[reverse]
    return x + 10000
