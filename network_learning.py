import numpy as np
import matplotlib.pyplot as plt

def make_matrix(N):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter
    Designed with 2 cliques representing visual/audio cortices
    return edge list
    '''
    roll = lambda p: int(np.random.uniform()<p)
    
    half_n = int(N/2)
    p1 = 0.5 # probability of connecting to node in clique
    p2 = 0.01 # probability of connecting to node outside clique
    w = 1/np.sqrt(N) # initial weight between connections
    A = np.zeros((N,N))
    E = []
    for i in range(N):
        for j in range(N):
            if (i<=half_n and j<=half_n) or (i>half_n and j>half_n):
                A[i,j] = roll(p1)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
            else:
                A[i,j] = roll(p2)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
                
    np.fill_diagonal(A, 0) # no self loops
    W = A*w
    return W,E

def make_fire_rate(N,nu1,nu2):
    '''
    Initializes the firing rate based on normal dis
    '''
    fire = {}
    for i in range(int(N/2)):
        fire[i] = max(0,np.random.normal(nu1))
        fire[i+int(N/2)] = max(0,np.random.normal(nu2))
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
        returns the new synaptic weight at time t+dt
    '''
    dw = -w+gamma*(nu1*nu2-w*nu1**2)
    return w+dt*dw


def bcm_step(eta,nu1,nu2,w,dt,theta,tau_t):
    '''
    BCM learning rule step
    '''
    dw = eta*nu1*nu2*(nu2-theta)
    dtheta = (1/tau_t)*(nu2**2 - theta)
    
    return w + dt*dw, theta + dtheta


def fire_step(tau_r,nu_d,h,w,dt):
    '''
    updates the firing rates
    '''
    N = len(nu_d.keys())
    nu = np.zeros((N,1))
    h = np.reshape(h,(len(h),1))
    for i in range(N):
        nu[i] = nu_d[i]
    out_nu = {}
    nu = nu+dt/tau_r*(-nu+(h+w@nu))
 
    for i in range(N):
        out_nu[i] = max(0,min(nu[i],3))
    return out_nu

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
        
        E[i][2] = w_tmp
        try:
            W[link[0],link[1]] = w_tmp
        except ValueError:
            print('w',w_tmp)
            print('f[link[0]]',f[link[0]])
            print('f[link[1]]',f[link[1]])
            print('link[2]',link[2])
    return W,E,theta
        

if __name__ == "__main__":
    # parameters
    N = 50       # network size
    nu1 = 0      # firing rate of 1st clique
    nu2 = 0      # firing rate of 2nd clique
    gamma = 0.1  # learning rate
    
    '''For BCM'''
    theta = 0 #strength threshold
    tau_t = 10 #threshold time update constant
    '''-------'''
    
    tau_r = 2 # firing rate update time constant
    tau_m = 2 # input update time constant
    # input voltage the node experiences at time 0
    h1 = 0   
    h2 = 0   
    # Input current 
    I1 = 1   
    I2 = 1.5 
    R = 1 # Resistance
    # time parameters
    T = 20           # end time
    dt = 0.01        #time step
    nt = int(T/dt)+1 # number of time steps

    # Combine parameters into a single vector for each node
    I = np.append(np.repeat(I1,N/2),np.repeat(I2,N/2)) 
    h = np.append(np.repeat(h1,N/2),np.repeat(h2,N/2))
    
    half_n = int(N/2)
    
    W,E = make_matrix(N) # returns weighted adjacency matrix, and edge list
    f = make_fire_rate(N,nu1,nu2)
    w_avg_1 =  np.zeros((nt,1))
    w_avg_2 = np.zeros((nt,1))
    h_avg = np.zeros((nt,1))
    f_avg = np.zeros((nt,1))
    f_avg2 = np.zeros((nt,1))
    w_avg_1[0] = np.mean(W[:half_n,:half_n])
    w_avg_2[0] = np.mean(W[half_n:N,half_n:N])
    h_avg[0] = np.mean(h)
    f_avg[0] = np.array(list(f.values())).mean()
    f_avg2[0] = np.array(list(f.values())).mean()
    
    weight_update_rules = ['oja', 'bcm']
    
    for i in range(1,nt):
        W_tmp,E,theta = learn(weight_update_rules[1],E,W[:,:],f,gamma,dt,theta,tau_t)
        f_tmp = fire_step(tau_r,f,h,W,dt)
        h = inpt_step(tau_m,h,R,I,dt)
        if i>=nt/20:
            I[:half_n]=0
        
        W = W_tmp
        f = f_tmp
        w_avg_1[i] = np.mean(W[:half_n,:half_n])
        w_avg_2[i] = np.mean(W[half_n:,half_n:])
        h_avg[i] = np.mean(h)
        f_avg[i] = np.array(list(f.values())[0:half_n]).mean()
        f_avg2[i] = np.array(list(f.values())[half_n:]).mean()

    t_ls = np.linspace(0,T,nt)
    fig = plt.figure()
    plt.plot(t_ls,w_avg_1,label='$w_1$')
    plt.plot(t_ls,w_avg_2,label='$w_2$')
    plt.plot(t_ls,h_avg,label='$h$')
    plt.plot(t_ls,f_avg,label='$\\nu_1$')
    plt.plot(t_ls,f_avg2,label='$\\nu_2$')
    plt.legend(loc='upper right')
    plt.show()
