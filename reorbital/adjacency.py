import numpy
#
# dij = Jij/Sqrt[Jii*Jjj] - 1.0
#
def distanceMatrix(eri):
    nb = eri.shape[0]
    dij = numpy.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            dij[i, j] = eri[i, i, j, j] / numpy.sqrt(
                eri[i, i, i, i] * eri[j, j, j, j])
    return dij


# Kij = (ij|ij)
def exchangeMatrix(eri):
    nb = eri.shape[0]
    kij = numpy.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            kij[i, j] = eri[i, j, i, j]
    return kij


# L = D - K
def laplacian(dij):
    nb = dij.shape[0]
    lap = numpy.zeros((nb, nb))
    lap = -dij
    # See DMRG in practive 2015 Dii = sum_j Kij
    diag = numpy.einsum('ij->i', dij)
    lap += numpy.diag(diag)
    return lap
