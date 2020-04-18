import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

np.random.seed(seed=1)

def make_matrix(c_size):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter
    Designed with 3 cliques representing visual/audio cortices
    return edge list
    '''
    roll = lambda p: int(np.random.uniform()<p)
    N = c_size*3
    p1 = 0.5 # probability of connecting to node in clique
    p2 = 0.01 # probability of connecting between sensory clique
    p3 = 0.1 # probability of connecting to input sensory clique to motor clique
    w = 1/np.sqrt(N) # initial weight between connections
    A = np.zeros((N,N))
    E = []
    for i in range(N):
        for j in range(N):
            # inside clique
            if (i<c_size and j<c_size) or (i>=c_size and j>=c_size and i<2*c_size and j<2*c_size) or (i>=2*c_size and j>=2*c_size):
                A[i,j] = roll(p1)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
            elif (i<2*c_size and j>=2*c_size) or (j<2*c_size and i>=2*c_size):
                A[i,j] = roll(p3)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
            else:
                A[i,j] = roll(p2)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
                
    np.fill_diagonal(A, 0) # no self loops
    W = A*w
    return W,E

def make_fire_rate(N,nu1,nu2,nu3):
    '''
    Initializes the firing rate based on normal dis
    '''
    stddev = 1 # Standard deviation of firing rate (mean is nu1, nu2)
    
    min_val = 0 # Minimum firing rate
    max_val = None # Maximum firing rate
    
    fire1 = np.clip(np.random.normal(loc=nu1,scale=stddev,size=int(N)),a_min=min_val,a_max=max_val)
    fire2 = np.clip(np.random.normal(loc=nu2,size=int(N)),a_min=min_val,a_max=max_val)
    fire3 = np.clip(np.random.normal(loc=nu2,size=int(N)),a_min=min_val,a_max=max_val)

    return np.hstack((fire1,fire2,fire3))

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
        returns the new synaptic weight at time t+dt
    '''
    dw = -w+gamma*(nu1*nu2-w*nu1**2)
    return w+dt*dw

def bcm_step(eta,nu1,nu2,w,dt,theta,tau_t):
    '''
    BCM learning rule step
    '''
    sigmoid = lambda x: 1/(1+np.exp(-x))
    dw = sigmoid(eta*nu1*nu2*(nu2-theta))
    dtheta = (1/tau_t)*(nu2**2 - theta)
    
    return w + dt*dw, theta + dtheta

def cov_step(eta,nu1,nu2,w,dt,nuk):
    '''
    Covariance update rule.
    '''
    nhalf = int(len(nuk)/2)
    mean_nu1, mean_nu2 = np.mean(nuk[:nhalf]), np.mean(nuk[nhalf:])
    dw = eta*(nu1-mean_nu1)*(nu2-mean_nu2)
    return w + dt*dw

def pattern_step(eta,nu1,nu2,w,dt):
    '''
    Pattern update rule. 
    This is the simplest but most time efficient weight update rule.
    '''
    dw = eta*nu1*(nu2-w)
    return w + dt*dw


def fire_step(tau_r,nu,h,w,dt):
    '''
    updates the firing rates
    '''
    sigmoid = lambda x: 1/(1+np.exp(-x))
    N = len(h)
    h_ = np.reshape(h,(N,1))
    nu_ = np.reshape(nu,(N,1))
    raw_out = np.reshape(sigmoid(w@nu),(N,1))
    
    nu = nu_+(dt/tau_r)*((-1)*nu_+(h_+raw_out))
    
    return np.clip(nu,0,3)

def inpt_step(tau_m,h,R,I,dt):
    '''
    Update the input voltage on nodes
    '''
    return h + dt/tau_m*(-h+R*I)

def learn(rule,E,W,f,gamma,dt,theta,tau_t):
    '''
    update the weights given a rule (BCM or Oja)
    '''
    N = len(E)
    for i in range(N):
        link = E[i]
        if rule=='oja':
            w_tmp = oja_step(gamma,f[link[0]],f[link[1]],link[2],dt)
        elif rule=='bcm':
            w_tmp, theta = bcm_step(gamma,f[link[0]],f[link[1]],link[2],dt,theta,tau_t)
        elif rule=='cov':
            w_tmp, theta = cov_step(gamma,f[link[0]],f[link[1]],link[2],dt,theta,tau_t,f)
        elif rule=='pattern':
            w_tmp, theta = pattern_step(gamma,f[link[0]],f[link[1]],link[2],dt,theta,tau_t)
        
        E[i][2] = w_tmp
        try:
            W[link[0],link[1]] = w_tmp
        except ValueError:
            print('w',w_tmp)
            print('f[link[0]]',f[link[0]])
            print('f[link[1]]',f[link[1]])
            print('link[2]',link[2])
    return W,E,theta


def neuron_weight_plot(weights, individual=-1, save_dir='.\\figures\\'):
    '''
    Parameters:
        - weights: (timestep, N, N) array of weights from which to slice and plot.
        - individual: optional integer, for plotting only one neuron slice.
        - save: string, if not empty, saves the plots.
    
    '''
    if individual>=0:
        fig = plt.figure()
        plt.plot(t_ls,weights[:,individual])
        plt.title('Neuron #'+f'{individual+1:03}')
        plt.show()
        if save_dir: fig.savefig(save_dir+'neuron_'+f'{individual+1:03}'+'_plot.png')
        return
    
    for i in range(weights.shape[1]):
            fig = plt.figure()
            plt.plot(t_ls,weights[:,i])
            plt.title('Neuron #'+f'{i+1:03}')
            plt.show()
            if save_dir: fig.savefig(save_dir+'neuron_'+f'{i+1:03}'+'_plot.png')
    return

def run_sim(W,E,tau_r,tau_m):
    # parameters
    c_size = 20  # size of each clique
    N = c_size*3 # network size
    nu1 = 0      # firing rate of 1st clique
    nu2 = 0      # firing rate of 2nd clique
    nu3 = 0      # firing rate of summing
    gamma = 0.1  # learning rate
    
    '''For BCM'''
    theta = 3 #strength threshold
    tau_t = 10 #threshold time update constant
    '''-------'''
    
    tau_r = 2 # firing rate update time constant
    tau_m = 2 # input update time constant
    # input voltage the node experiences at time 0
    h1 = 0   
    h2 = 0   
    h3 = 0 
    # Input current 
    I1 = 1   
    I2 = 1.5
    I3 = 0 
    R = 1 # Resistance
    # time parameters
    T = 10           # end time
    dt = 0.1         #time step
    nt = int(T/dt)+1 # number of time steps

    # Combine parameters into a single vector for each node
    I = np.concatenate((np.repeat(I1,c_size),np.repeat(I2,c_size),np.repeat(I3,c_size))) 
    h = np.concatenate((np.repeat(h1,c_size),np.repeat(h2,c_size),np.repeat(h3,c_size)))
        
    # W,E = make_matrix(c_size) # returns weighted adjacency matrix, and edge list
    f = make_fire_rate(c_size,nu1,nu2,nu3)

    output = np.zeros((nt,1))
    weight_update_rules = ['oja', 'bcm']

    for i in range(nt):
        if i>=nt/10:
            I[:c_size]=0
        
        W,E,theta = learn(weight_update_rules[1],E,W[:,:],f,gamma,dt,theta,tau_t)
        f = fire_step(tau_r,f,h,W,dt)
        h = inpt_step(tau_m,h,R,I,dt)
        
        output[i] = np.sum(W[2*c_size:,2*c_size:]@f[2*c_size:])
    return output[-1][0]

if __name__ == "__main__":
    W,E = make_matrix(20) 
    grid_num = 20
    # tau_m
    tau_m_min = 0.5
    tau_m_max = 5
    dtm = (tau_m_max-tau_m_min)/grid_num
    tau_m_range =  np.arange(tau_m_min, tau_m_max+dtm, dtm)
    #tau_r
    tau_r_min = 0.5
    tau_r_max = 5
    dtr = (tau_r_max-tau_r_min)/grid_num
    tau_r_range =  np.arange(tau_r_min, tau_r_max+dtr, dtr)

    x,y =  np.mgrid[slice(tau_r_min, tau_r_max+dtr, dtr),
                    slice(tau_m_min, tau_m_max+dtm, dtm)]
    z = []
    n=grid_num**2
    cnt = 0
    for i in tau_r_range:
        for j in tau_m_range:
            # start = time.time()
            cnt+=1
            o = run_sim(W,E,i,j)
            z.append(o)
            if not cnt%100:
                print(cnt/n)
            # end = time.time()
            # print(end - start)
            # print((end - start)*n)
            # exit()
    z = np.reshape(z, (x.shape))
    z_min, z_max = z.min(), z.max()
    c = plt.pcolormesh(x,y,z, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.xlabel('$\\tau_r$')
    plt.ylabel('$\\tau_m$')
    plt.colorbar(c)
    plt.show()