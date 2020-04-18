import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import signal

np.random.seed(seed=1)

def make_matrix(c_size):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter
    Designed with 3 cliques representing visual/audio cortices
    return edge list
    '''
    roll = lambda p: int(np.random.uniform()<p)
    N = c_size*3
    p1 = 0.6 # probability of connecting to node in clique
    p2 = 0.1 # probability of connecting between sensory clique
    p3 = 0.3 # probability of connecting to input sensory clique to motor clique
    w = 1/np.sqrt(N) # initial weight between connections
    A = np.zeros((N,N))
    E = []
    for i in range(N):
        for j in range(N):
            # inside clique
            if (i<c_size and j<c_size) or (i>=c_size and j>=c_size and i<2*c_size and j<2*c_size) or (i>=2*c_size and j>=2*c_size):
                A[i,j] = roll(p1)*w
                if A[i,j] and i!=j:
                    E.append([i,j,w])
            elif (i<2*c_size and j>=2*c_size) or (j<2*c_size and i>=2*c_size):
                A[i,j] = roll(p3)
                # if A[i,j] and i!=j:
                #     E.append([i,j,1])
            else:
                A[i,j] = roll(p2)*w
                if A[i,j] and i!=j:
                    E.append([i,j,w])
                
    np.fill_diagonal(A, 0) # no self loops
    return A,E

def bcm(w,eta,nutheta,nu,u,dt):
    '''
    BCM rule with a changing nu_theta where it is passed through a low-pass filter
    '''
    return w+dt*(eta*nu*(nu-nutheta)*u), nutheta+dt*(nu**2-nutheta)

def intandfire(u,urest,taum,R,I,uth):
    '''
    Leaky integrate and fire
    '''
    spiked = 0
    newu = u + dt/taum*(-u+urest+R*I+w@u)
    if u>=uth:
        newu=0
        spiked = 1
    return newu,spiked

def init_input(I1,I2,I3,n):
    I1_l = np.clip(np.random.uniform(I1-0.5,I1+0.5,size=n),0,None)
    I2_l = np.clip(np.random.uniform(I2-0.5,I2+0.5,size=n),0,None)
    I3_l = np.clip(np.random.uniform(I3-0.5,I3+0.5,size=n),0,None)
    return np.hstack((I1_l,I2_l,I3_l))

