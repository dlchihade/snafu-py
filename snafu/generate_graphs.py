from . import *

# See Steyvers & Tenenbaum (2005)
# Include an extra parameter 'tail' which allows m-1 "null" nodes in
# neighborhood of every node to better match scale-free distribution.
def generate_tenenbaum_steyvers_network(n, m, tail=True, seed=None):
    """
    Generate a scale-free network using the Steyvers & Tenenbaum (2005) model.

    Constructs a network where each new node attaches to a randomly chosen
    neighborhood of an existing node, optionally padded with null (non-attaching)
    nodes to simulate a power-law distribution in node degrees.

    Parameters
    ----------
    n : int
        Total number of nodes in the graph.
    m : int
        Number of edges each new node forms. Must be <= n.
    tail : bool, optional
        If True, adds (m-1) null nodes to each candidate neighborhood to allow
        non-attachment and better simulate a scale-free structure. Default is True.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    networkx.Graph
        An undirected graph generated according to the Steyvers-Tenenbaum model.
    """


    nplocal = np.random.RandomState(seed)
    a = np.zeros((n,n))                                  # initialize matrix
    for i in range(m):                                   # complete m x m graph
        for j in range(m):
            if i != j:
                a[i,j] = 1
    for i in range(m,n):                                 # for the rest of nodes, preferentially attach
        nodeprob = sum(a) / sum(sum(a))                  # choose node to differentiate with this probability distribution
        diffnode = nplocal.choice(n,p=nodeprob)          # node to differentiate
        h = list(np.where(a[diffnode])[0]) + [diffnode]  # neighborhood of diffnode
        if tail==True:
            h = h + [-1] * (m-1)
        #hprob=sum(a[:,h])/sum(sum(a[:,h]))              # attach proportional to node degree?
        #tolink=nplocal.choice(h,m,replace=False,p=hprob)
        tolink = nplocal.choice(h,m,replace=False)       # or attach randomly
        for j in tolink:
            if j != -1:
                a[i,j] = 1
                a[j,i] = 1
    return nx.to_networkx_graph(a)
