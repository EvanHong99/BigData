cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def update_rank(dict out_links,np.ndarray[double,ndim=1] new_rank,np.ndarray[double,ndim=1] old_rank,double alpha):
    cdef int node, degree, link
    for node, [degree, links] in out_links.items():
        for link in links:
            new_rank[link - 1] += alpha * old_rank[node - 1] / degree
    return new_rank