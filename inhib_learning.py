import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def make_matrix(N,Ni):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter
    Designed with 2 cliques representing visual/audio cortices
    return edge list
    '''
    roll = lambda p: int(np.random.uniform()<p)
    Nt = N+Ni
    half_n = int((N)/2)
    p1 = 0.6 # probability of connecting to node in clique
    p2 = 0.1 # probability of connecting to node outside clique
    p3 = 0.3 # probability of connecting to node outside clique
    w = 1/np.sqrt(N) # initial weight between connections
    A = np.zeros((Nt,Nt))
    E = []
    for i in range(Nt):
        for j in range(Nt):
            if (i<=half_n and j<=half_n) or (i>half_n and j>half_n and i<=N and j<=N):
                A[i,j] = roll(p1)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
            elif (i>N and j>N):
                A[i,j] = roll(p1)/(2*w)
            elif (i>N and j<=N) or (i<=N and j>N):
                A[i,j] = -roll(p1)/(2*w)
            else:
                A[i,j] = roll(p2)
                if A[i,j] and i!=j:
                    E.append([i,j,w])
                
    np.fill_diagonal(A, 0) # no self loops
    W = A*w
    return W,E

def make_fire_rate(N,Ni,nu1,nu2,nui):
    '''
    Initializes the firing rate based on normal dis
    '''
    stddev = 1 # Standard deviation of firing rate (mean is nu1, nu2)
    
    min_val = 0 # Minimum firing rate
    max_val = None # Maximum firing rate
    
    fire1 = np.clip(np.random.normal(loc=nu1,scale=stddev,size=int(N/2)),a_min=min_val,a_max=max_val)
    fire2 = np.clip(np.random.normal(loc=nu2,size=int(N/2)),a_min=min_val,a_max=max_val)
    fire3 = np.clip(np.random.normal(loc=nui,size=int(Ni)),a_min=min_val,a_max=max_val)
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
    dw = eta*nu1*nu2*(nu2-theta)
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
    raw_out = np.reshape(sigmoid((w@nu_)+h_),(N,1))

    nu = nu_+(dt/tau_r)*(-nu_+raw_out)

    return nu

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


def input_manipulation(rule,I,i,nt,shutoff_time):
    if rule=='fixed shutoff':
        if i>=nt*shutoff_time:
            I[:half_n]=0
    elif rule=='sin':
        pass
    elif rule=='cos':           
        pass
    
    return I


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


if __name__ == "__main__":
    # parameters
    N = 60       # network size
    Ni = 20
    Nt = N+Ni
    nu1 = 0      # firing rate of 1st clique
    nu2 = 0      # firing rate of 2nd clique
    nui = 1      # firing rate of 2nd clique
    gamma = 0.1  # learning rate
    
    '''For BCM'''
    theta = 0.5 #strength threshold
    tau_t = 1 #threshold time update constant
    '''-------'''
    
    tau_r = 2 # firing rate update time constant
    tau_m = 2 # input update time constant
    # input voltage the node experiences at time 0
    h1 = 0   
    h2 = 0 
    hi = 0  
    # Input current 
    I1 = 10   
    I2 = 9
    Ii = 1
    shutoff_time = 0.25 # % of the way through training when input is shut off.
    
    R = 1 # Resistance
    # time parameters
    T = 30           # end time
    dt = 0.01        #time step
    nt = int(T/dt)+1 # number of time steps

    # Combine parameters into a single vector for each node
    I = np.concatenate((np.repeat(I1,N/2),np.repeat(I2,N/2),np.repeat(Ii,Ni))) 
    h = np.concatenate((np.repeat(h1,N/2),np.repeat(h2,N/2),np.repeat(hi,Ni)))
    
    half_n = int(N/2)
    
    W,E = make_matrix(N,Ni) # returns weighted adjacency matrix, and edge list
    f = make_fire_rate(N,Ni,nu1,nu2,nui)
    w_avg_1 =  np.zeros((nt,1))
    w_avg_2 = np.zeros((nt,1))
    w_avg_3 =  np.zeros((nt,1))
    w_avg_4 = np.zeros((nt,1))
    h_avg = np.zeros((nt,1))
    f_avg = np.zeros((nt,1))
    f_avg2 = np.zeros((nt,1))
    
    #all_weights = np.zeros((nt,N,N))
    output = np.zeros((nt,1))
    
    weight_update_rules = ['oja', 'bcm', 'cov', 'pattern']
    input_rules = ['fixed shutoff','sin','cos']

    # Rate to print graphml files
    output_gephi = False
    graph_steps = list(np.arange(1,nt,int(nt/10)))
    graph_steps.append(int(nt/20)-1)
    graph_steps.append(int(nt/20)+3)
    nz = len(str(nt))
    for i in range(nt):
        print ('\rCompletion: ' + f'{round(((i+1)/nt)*100,2):0>2}', end='')
        
        I = input_manipulation(input_rules[0],I,i,nt,shutoff_time)
        
        W,E,theta = learn(weight_update_rules[1],E,W[:,:],f,gamma,dt,theta,tau_t)
        f = fire_step(tau_r,f,h,W,dt)
        h = inpt_step(tau_m,h,R,I,dt)
        
        #all_weights[i] = W
        w_avg_1[i] = W[:half_n,:half_n][np.nonzero(W[:half_n,:half_n])].mean() 
        w_avg_2[i] = W[half_n:N,half_n:N][np.nonzero(W[half_n:N,half_n:N])].mean() 
        w_avg_3[i] = W[:half_n,half_n:N][np.nonzero(W[:half_n,half_n:N])].mean() 
        w_avg_4[i] = W[half_n:N,:half_n][np.nonzero(W[half_n:N,:half_n])].mean() 
        h_avg[i] = np.mean(h)
        f_avg[i] = f[:half_n].mean()
        f_avg2[i] = f[half_n:N].mean()
        
        output[i] = np.sum(W[0:N,0:N]@f[:N])

        # Gephi output
        if output_gephi and i in graph_steps:
            G = nx.from_numpy_matrix(W)
            step = str(i).zfill(nz)
            nx.write_graphml(G,'bcm_run/step_{}_adj.graphml'.format(step))
    print('')
    
    t_ls = np.linspace(0,T,nt)
    fig, ax = plt.subplots(figsize=(12,7))
    plt.plot(t_ls,w_avg_1,label='$w_{1,1}$')
    plt.plot(t_ls,w_avg_2,label='$w_{1,2}$')
    plt.plot(t_ls,w_avg_3,label='$w_{2,1}$')
    plt.plot(t_ls,w_avg_4,label='$w_{2,2}$')
    plt.plot(t_ls,h_avg,label='$h$')
    plt.plot(t_ls,f_avg,label='$\\nu_1$')
    plt.plot(t_ls,f_avg2,label='$\\nu_2$')
    plt.legend(loc='upper right')
    
    ax2=ax.twinx()
    ax2.plot(t_ls,output,color='black',linewidth=2,label='output')
    
    plt.legend(loc='upper right')
    plt.show()
    
    #neuron_weight_plot(all_weights,save_dir='')