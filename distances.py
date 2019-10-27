import numpy as np
import ot


def wmd(p, q, C, truncate=None):
    """ Word mover's distance between distributions p and q with cost M."""
    if truncate is None:
        return ot.emd2(p, q, C)
    id_p = np.argsort(p)[-truncate:]
    id_q = np.argsort(q)[-truncate:]
    C_reduced = C[id_p][:, id_q]
    return ot.emd2(p[id_p], q[id_q], C_reduced)


def rwmd(p, q, C):
    """ Relaxed WMD between distributions p and q with cost M."""
    active1 = np.where(p)[0]
    active2 = np.where(q)[0]
    C_reduced = C[active1][:, active2]
    l1 = (np.min(C_reduced, axis=1) * p[active1]).sum()
    l2 = (np.min(C_reduced, axis=0) * q[active2]).sum()
    return max(l1, l2)


def wcd(p, q, embeddings):
    """ Word centroid distance between p and q under embeddings."""
    m1 = np.mean(embeddings.T * p, axis=1)
    m2 = np.mean(embeddings.T * q, axis=1)
    return np.linalg.norm(m1 - m2)


def prefetch_and_prune(query, docs, embeddings, C, k):
    dists = [wcd(query, doc, embeddings) for doc in docs]
    costs = sorted([(i, dist) for i, dist in enumerate(dists)],
                   key=lambda x: x[1])
    wmds = []
    for i in range(k):
        idx = costs[i][0]
        wmds.append((idx, ot.emd2(query, docs[idx], C)))
    max_wmd = max([wmd[1] for wmd in wmds])
    for i in range(k, len(costs)):
        idx = costs[i][0]
        rwm = rwmd(query, docs[idx], C)
        if rwm < max_wmd:
            wmds.append((idx, ot.emd2(query, docs[idx], C)))
    top_k = sorted(wmds, key=lambda x: x[1])

    return [i for (i, dist) in top_k]
