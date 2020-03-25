import numpy as np
import matplotlib.pyplot as plt

def roll(p):
    if np.random.uniform()<p:
        return 1
    return 0

def make_matrix(N):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter
    Designed with 2 cliques representing visual/audio cortices
    return edge list
    '''
    half_n = int(N/2)
    p1 = 0.7
    p2 = 0.2
    w = 0.5
    A = np.zeros((N,N))
    E = []
    for i in range(N):
        for j in range(N):
            if (i<=half_n and j<=half_n) or (i>half_n and j>half_n):
                A[i,j] = roll(p1)
                if A[i,j] and i!=j:
                #if i!=j and roll(p1):
                    E.append([i,j,w])
            else:
                A[i,j] = roll(p2)
                if A[i,j] and i!=j:
                #if i!=j and roll(p2):
                    E.append([i,j,w])
                
    np.fill_diagonal(A, 0)
    W = A*0.5
    return W,E

def make_fire_rate(N,nu1,nu2):
    '''
    
    '''
    fire = {}
    for i in range(int(N/2)):
        fire[i] = np.random.normal(nu1)
        fire[i+int(N/2)] = np.random.normal(nu2)
    return fire

def oja_step(gamma,nu1,nu2,w,dt):
    '''
    Oja learning rule solved with forward Euler
    inpt:
        gamma - learning rate
        nu1 - firing rate of the pre synaptic neuron
        nu2- firing rate of the post synaptic neuron
        w - synaptic weight at time t
        dt - time step
    oupt:
        returns the new synaptic weight at time t+dt, where there is a hard max at 10
    '''
    dw = gamma*(nu1*nu2-w*nu1**2)
    return min(w+dt*dw,1000) # just in case the weight blows up we cut if off at 10

def learn(E,W,f,gamma,dt):
    N = len(E)
    for i in range(N):
        link = E[i]
        w_tmp = oja_step(gamma,f[link[0]],f[link[1]],link[2],dt)
        E[i][2] = w_tmp
        W[link[0],link[1]] = w_tmp
    return W,E
        

if __name__ == "__main__":
    # parameters
    N = 50       # network size
    nu1 = 10     # firing rate of 1st clique
    nu2 = 13     # firing rate of 2nd clique
    gamma = 0.4  # learning rate
    T = 100
    dt = 0.1
    nt = int(T/dt)+1

    W0,E = make_matrix(N)
    f = make_fire_rate(N,nu1,nu2)
    f2 = make_fire_rate(N,0,nu2)
    W = np.zeros((N,N,nt))
    W[:,:,0] = W0
    for i in range(1,nt):
        W[:,:,i],E = learn(E,W[:,:,i-1],f,gamma,dt)
        if i>nt/2:
            f = f2


