import ot


def hott(p, q, C, threshold=None):
    """ Hierarchical optimal topic transport."""
    k = len(p)
    if threshold is None:
        threshold = 1. / (k + 1)
    id_p = p > threshold
    id_q = q > threshold
    C_reduced = C[id_p][:, id_q]
    return ot.emd2(p[id_p], q[id_q], C_reduced)


def hoftt(p, q, C):
    """ Hierarchical optimal full topic transport."""
    return ot.emd2(p, q, C)
