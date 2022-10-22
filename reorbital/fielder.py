import numpy
from scipy.linalg import eigh

# Get the orbital ordering
def fielder(eri, mode='kij', debug=False):
    from reorbital.adjacency import laplacian
    if debug:
        print('\n[fielder.orbitalOrdering] determing ordering based on', mode.lower(
        ))
    nb = eri.shape[0]
    if mode.lower() == 'dij':
        from reorbital.adjacency import distanceMatrix
        dij = distanceMatrix(eri)
    elif mode.lower() == 'kij':
        from reorbital.adjacency import exchangeMatrix
        dij = exchangeMatrix(eri)
    elif mode.lower() == 'kmat':
        dij = eri.copy()
    lap = laplacian(dij)
    eig, v = eigh(lap)
    # From postive to negative
    order = numpy.argsort(v[:, 1])[::-1]
    if debug:
        print('dij:\n', dij)
        print('eig:\n', eig)
        print('v[1]=', v[:, 1])
        print('new order:', order)
    return order
