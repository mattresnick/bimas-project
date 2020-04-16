import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
    fire = np.zeros((N*3,1))
    for i in range(N):
        fire[i] = max(0,np.random.normal(nu1))
        fire[i+N] = max(0,np.random.normal(nu2))
        fire[i+2*N] = max(0,np.random.normal(nu3))
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


def fire_step(tau_r,nu,h,w,dt):
    '''
    updates the firing rates
    '''
    #N = len(nu_d.keys())
    #nu = np.zeros((N,1))
    h = np.reshape(h,(len(h),1))
    # for i in range(N):
    #     nu[i] = nu_d[i]
    # out_nu = {}
    out_nu = np.zeros((len(nu),1))
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
    T = 20           # end time
    dt = 0.01        #time step
    nt = int(T/dt)+1 # number of time steps

    # Combine parameters into a single vector for each node
    I = np.concatenate((np.repeat(I1,c_size),np.repeat(I2,c_size),np.repeat(I3,c_size))) 
    h = np.concatenate((np.repeat(h1,c_size),np.repeat(h2,c_size),np.repeat(h3,c_size)))
        
    W,E = make_matrix(c_size) # returns weighted adjacency matrix, and edge list
    f = make_fire_rate(c_size,nu1,nu2,nu3)
    w_avg_1 =  np.zeros((nt,1))
    w_avg_2 = np.zeros((nt,1))
    w_avg_3 =  np.zeros((nt,1))
    w_avg_4 = np.zeros((nt,1))
    w_avg_5 = np.zeros((nt,1))
    w_avg_6 = np.zeros((nt,1))
    w_avg_7 = np.zeros((nt,1))
    w_avg_8 = np.zeros((nt,1))
    w_avg_9 = np.zeros((nt,1))
    h_avg = np.zeros((nt,1))
    f_avg = np.zeros((nt,1))
    f_avg2 = np.zeros((nt,1))
    f_avg3 = np.zeros((nt,1))
    
    all_weights = np.zeros((nt,N,N))
    output = np.zeros((nt,1))
    weight_update_rules = ['oja', 'bcm']

    ## Gephi
    output_gephi = False
    graph_steps = list(np.arange(1,nt,int(nt/10)))  # Rate to print graphml files
    graph_steps.append(int(nt/20)-1)
    graph_steps.append(int(nt/20)+3)
    nz = len(str(nt))

    for i in range(nt):
        if i>=nt/20:
            I[:c_size]=0
        
        W,E,theta = learn(weight_update_rules[1],E,W[:,:],f,gamma,dt,theta,tau_t)
        f = fire_step(tau_r,f,h,W,dt)
        h = inpt_step(tau_m,h,R,I,dt)
        
        all_weights[i] = W
        w_avg_1[i] = np.mean(W[:c_size,:c_size])                 # Intracortical averages 1 (Quadrant 1)
        w_avg_5[i] = np.mean(W[c_size:2*c_size,c_size:2*c_size]) # Intracortical averages 5 (Quadrant 5)
        w_avg_9[i] = np.mean(W[2*c_size:,2*c_size:])             # Intracortical averages 9 (Quadrant 9)
        w_avg_2[i] = np.mean(W[:c_size,c_size:2*c_size])         # Intercortical averages 2 (Quadrant 2)
        w_avg_3[i] = np.mean(W[:c_size,2*c_size:])               # Intercortical averages 3 (Quadrant 3)
        w_avg_4[i] = np.mean(W[c_size:2*c_size,:c_size])         # Intercortical averages 4 (Quadrant 4)
        w_avg_6[i] = np.mean(W[c_size:2*c_size,2*c_size:])       # Intercortical averages 6 (Quadrant 6)
        w_avg_7[i] = np.mean(W[2*c_size,:c_size])                # Intercortical averages 7 (Quadrant 7)
        w_avg_8[i] = np.mean(W[2*c_size:,c_size:2*c_size])       # Intercortical averages 8 (Quadrant 8)
        h_avg[i] = np.mean(h)
        f_avg[i] = f[0:c_size].mean()
        f_avg2[i] = f[c_size:2*c_size].mean()
        f_avg3[i] = f[2*c_size:].mean()
        # f_avg2[i] = np.array(list(f.values())[c_size:2*c_size]).mean()
        # f_avg3[i] = np.array(list(f.values())[2*c_size:]).mean()
        
        output[i] = np.sum(W[2*c_size:,2*c_size:]@f[2*c_size:])

        # Gephi output
        if output_gephi and i in graph_steps:
            G = nx.from_numpy_matrix(W)
            step = str(i).zfill(nz)
            nx.write_graphml(G,'bcm_run/step_{}_adj.graphml'.format(step))

    t_ls = np.linspace(0,T,nt)
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(t_ls,w_avg_1,label='$w_{1,1}$')
    ax[0,0].plot(t_ls,w_avg_5,label='$w_{2,2}$')
    ax[0,0].plot(t_ls,w_avg_9,label='$w_{3,3}$')
    ax[0,0].legend(loc='upper left')
    ax[0,0].set_title('Intracortical Weights')

    ax[1,0].plot(t_ls,w_avg_2,label='$w_{1,2}$')
    ax[1,0].plot(t_ls,w_avg_3,label='$w_{1,3}$')
    ax[1,0].plot(t_ls,w_avg_4,label='$w_{2,1}$')
    ax[1,0].plot(t_ls,w_avg_6,label='$w_{2,3}$')
    ax[1,0].plot(t_ls,w_avg_7,label='$w_{3,1}$')
    ax[1,0].plot(t_ls,w_avg_8,label='$w_{3,2}$')
    ax[1,0].legend(loc='upper left')
    ax[1,0].set_title('Intercortical Weights')
    
    #plt.plot(t_ls,h_avg,label='$h$')
    ax[0,1].plot(t_ls,f_avg,label='$\\nu_1$')
    ax[0,1].plot(t_ls,f_avg2,label='$\\nu_2$')
    ax[0,1].plot(t_ls,f_avg3,label='$\\nu_3$')
    ax[0,1].legend(loc='upper right')
    ax[0,1].set_title('Firing Rate Averages In Each Cortex')
    
    #ax2=ax.twinx()
    ax[1,1].plot(t_ls,output,color='black',linewidth=2,label='output') 
    ax[1,1].legend(loc='upper left')
    ax[1,1].set_title('Output')
    plt.show()
    
    #neuron_weight_plot(all_weights,save_dir='')
