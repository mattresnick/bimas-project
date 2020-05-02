import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import GlobalRules

def make_matrix(N, intra_prob=0.7, inter_prob = 0.25):
    '''
    Make an adjacency matrix with higher intra connectivity density than inter.
    Designed with 2 cliques representing visual/audio cortices.
    
    Paramters:
        - N: (int) Number of neurons in the entire population.
        - intra_prob: (float) probability for intracortical connection 0<p<1
        - inter_prob: (float) probability for intercortical connection 0<p<1, and should be less than intra_prob.
    
    Returns:
        - W: (2D ndarray) Weighted edge matrix for neuron population.
        - E: (2D ndarray) List of connection indices and corresponding weight values for efficient updates during learning.
    
    Where A=intra, E=inter, the weight regimes are structured as:
    
    A | E
    --+--
    E | A
    
    '''
    half_n = int(N/2)
    
    # Initize weights as uniform sampling from 0 to 1.
    W = np.random.uniform(0,1,(N,N)) 
    
    connection_masks = []
    for i in range(4):
        # Initialize random neuron connection values.
        W_connection = np.random.uniform(0,1,(half_n,half_n))
        
        # Make connection masks, with connection probability 0.5 or 0.05,
        # depending on the regime.
        prob = inter_prob
        if not i%3: prob = intra_prob
        connection_masks.append((W_connection<prob)*W_connection)
    
    # Construct the mask from all 4 regimes.
    top_half_mask = np.hstack((connection_masks[0],connection_masks[1]))
    bottom_half_mask = np.hstack((connection_masks[2],connection_masks[3]))
    full_mask = np.vstack((top_half_mask, bottom_half_mask))
    
    # Mask the weights and zero the diagonal to remove self-loops.
    W = full_mask
    np.fill_diagonal(W, 0)
    
    E = []
    I,J = W.shape
    for i in range(I):
        for j in range(J):
            E.append(np.array([i,j,W[i][j]]))
    
    return W, np.array(E)

def make_fire_rate(N,nu1,nu2):
    '''
    Initializes the firing rate based on normal dis
    '''
    stddev = 1 # Standard deviation of firing rate (mean is nu1, nu2)
    
    min_val = 0 # Minimum firing rate
    max_val = None # Maximum firing rate
    
    fire1 = np.clip(np.random.normal(loc=nu1,scale=stddev,size=int(N/2)),a_min=min_val,a_max=max_val)
    fire2 = np.clip(np.random.normal(loc=nu2,size=int(N/2)),a_min=min_val,a_max=max_val)
    
    return np.hstack((fire1,fire2))

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
    raw_out = np.reshape(sigmoid((w@nu_)*h_),(N,1))

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


def input_manipulation(rule,I,i,nt,shutoff_time,turnon_time,frequency=1):
    if rule=='fixed shutoff':
        if i>=nt*shutoff_time:
            I[:half_n]=0
            
    elif rule=='shutoff recovery':
        if i>=nt*shutoff_time and i<nt*turnon_time:
            I[:half_n]=0
        elif i>=nt*turnon_time:
            I[:half_n]=1
            
    elif rule=='sin':
        # I.e. start in the "on" state
        angle = (np.pi*2*i*frequency)/nt
        I[:half_n]=round((np.sin(angle)+1)/2)
    
    elif rule=='inv_sin':
        # Inverted sine, i.e. start in the "off" state
        angle = (np.pi*2*i*frequency)/nt
        I[:half_n]=round(((-1)*np.sin(angle)+1)/2)
        
    elif rule=='cos':
        # I.e. offset sine
        angle = (np.pi*2*i*frequency)/nt
        I[:half_n]=round((np.cos(angle)+1)/2)
    
    elif rule=='damped':
        angle = (np.pi*2*(1.0005**i)*frequency)/nt
        I[:half_n]=round((np.sin(angle)+1)/2)
    
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
    N = 40       # network size
    nu1 = 0      # firing rate of 1st clique
    nu2 = 0      # firing rate of 2nd clique
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
    # Input current 
    I1 = 10   
    I2 = 9
    shutoff_time = 0.25 # % of the way through training when input is shut off.
    turnon_time = 0.9 # % of the way through training when input is turned back on.
    
    R = 1 # Resistance
    # time parameters
    T = 30           # end time
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
    w_avg_3 =  np.zeros((nt,1))
    w_avg_4 = np.zeros((nt,1))
    h_avg = np.zeros((nt,1))
    f_avg = np.zeros((nt,1))
    f_avg2 = np.zeros((nt,1))
    num_aud_zeros = np.zeros((nt,1))
    
    all_weights = np.zeros((nt,N,N))
    output = np.zeros((nt,1))
    
    weight_update_rules = ['oja', 'bcm', 'cov', 'pattern']
    input_rules = ['fixed shutoff','shutoff recovery','sin','inv_sin','cos','damped']

    # Rate to print graphml files
    output_gephi = False
    graph_steps = list(np.arange(1,nt,int(nt/10)))
    graph_steps.append(int(nt/20)-1)
    graph_steps.append(int(nt/20)+3)
    nz = len(str(nt))
    global_rule = GlobalRules.Siphoning(N, free=True)
    
    for i in range(nt):
        print ('\rCompletion: ' + f'{round(((i+1)/nt)*100,2):0>2}', end='')
        
        I = input_manipulation(input_rules[1],I,i,nt,shutoff_time,turnon_time)
        
        W,E,theta = learn(weight_update_rules[1],E,W[:,:],f,gamma,dt,theta,tau_t)
        f = fire_step(tau_r,f,h,W,dt)
        h = inpt_step(tau_m,h,R,I,dt)
        
        all_weights[i] = W
        w_avg_1[i] = W[:half_n,:half_n][np.nonzero(W[:half_n,:half_n])].mean() #np.mean(W[:half_n,:half_n]) # Intracortical averages 1 (Quadrant 4)
        w_avg_2[i] = W[half_n:,half_n:][np.nonzero(W[half_n:,half_n:])].mean() #np.mean(W[half_n:,half_n:]) # Intracortical averages 2 (Quadrant 2)
        w_avg_3[i] = W[:half_n,half_n:][np.nonzero(W[:half_n,half_n:])].mean() #np.mean(W[:half_n,half_n:]) # Intercortical averages 1 (Quadrant 1)
        w_avg_4[i] = W[half_n:,:half_n][np.nonzero(W[half_n:,:half_n])].mean() #np.mean(W[half_n:,:half_n]) # Intercortical averages 2 (Quadrant 3)
        h_avg[i] = np.mean(h)
        f_avg[i] = f[:half_n].mean()
        f_avg2[i] = f[half_n:].mean()
        
        output[i] = np.sum(W@f)
        
        W = global_rule.run(f,W,i)
        
        aud_zeros = len(np.where(W[half_n:,half_n:]==0)[0])
        num_aud_zeros[i] = aud_zeros
        
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
    
    
    
    # Separated plots.
    fig, ax = plt.subplots(figsize=(12,7))
    plt.plot(t_ls,w_avg_1,label='$w_1$',color='red')
    plt.plot(t_ls,w_avg_2,label='$w_2$',color='yellow')
    plt.plot(t_ls,w_avg_3,label='$w_3$',color='blue')
    plt.plot(t_ls,w_avg_4,label='$w_4$',color='green')
    plt.legend(loc='upper right')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12,7))
    plt.plot(t_ls,h_avg,label='$h$',color='brown')
    plt.plot(t_ls,f_avg,label='$\\nu_1$',color='purple')
    plt.plot(t_ls,f_avg2,label='$\\nu_2$',color='black')
    plt.legend(loc='upper right')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12,7))
    plt.plot(t_ls,num_aud_zeros)
    plt.show()
    
    #neuron_weight_plot(all_weights,save_dir='')
