import numpy as np
import matplotlib.pyplot as plt

'''
The main differences here, besides the update rule itself, is how the data is created and streamed.

The input is comprised of 100 samples of input, each with two components (both are floating point
values sampled from the standard normal distribution), where the first represents visual input
and the second represents auditory.

There are N neurons, each with 2 weight values (or, this can be thought of as two cliques of N 
neurons), and the output of each is the dot product of the weights and a given input example. 
The neurons are trained by feeding every input individually to the network and updating the weights 
each time based on the the calculated output. This is done a specified amount of times, and halfway 
through, the visual input is set to zero.

The plots you see are the historical means of the absolute values of the weights for each clique.
Without turning off the input, they should converge to essentially the same spot.
'''

N = 26 # Number of neurons.
W = np.random.normal(0,1,(N,2))

X = np.random.normal(0,1,(100,2))
gamma = 0.01
time_steps = 100
dt = 1
turnoff_flag = True

save_w = [np.array([np.mean(np.abs(W[0])),np.mean(np.abs(W[1]))])]
for step in range(time_steps):
    
    if step/time_steps>0.5 and turnoff_flag:
        X[:,0] = np.zeros(100)
        turnoff_flag = False
    
    delta_w = np.zeros((N,2))
    for i,x_i in enumerate(X):
        y = (X[i]@W.T)[:,np.newaxis] # (1,2) dot (2,N) --> (1,N)
        y_norm = y@y.T # --> (N,N)
        y_norm = np.tril(y_norm) 
        delta_w += (y@(X[i][:,np.newaxis]).T - y_norm@W)
    
    W = W + (1/dt)*gamma*delta_w
    W = W/np.linalg.norm(W, axis=1)[:,np.newaxis]
    dt+=1
    save_w.append(np.array([np.mean(np.abs(W[0])),np.mean(np.abs(W[1]))]))

x = list(range(len(save_w)))
save_w = np.array(save_w)

fig, ax = plt.subplots(figsize=(12,7))
plt.plot(x,save_w[:,0],label='$w_1$')
plt.plot(x,save_w[:,1],label='$w_2$')
plt.legend(loc='upper right')
plt.show()
