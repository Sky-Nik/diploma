import numpy.linalg as la

def norm(x):
    if hasattr(x, 'norm'):
        return x.norm()
    return la.norm(x)
