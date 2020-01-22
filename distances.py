import numpy as np
from hott import sparse_ot

def wmd(p, q, C, truncate=None):
    """ Word mover's distance between distributions p and q with cost M."""
    if truncate is None:
        return sparse_ot(p, q, C)
    
    # Avoid changing p and q outside of this function
    p, q = np.copy(p), np.copy(q)
    
    to_0_p_idx = np.argsort(-p)[truncate:]
    p[to_0_p_idx] = 0
    to_0_q_idx = np.argsort(-q)[truncate:]
    q[to_0_q_idx] = 0
    
    return sparse_ot(p, q, C)


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
