import numpy as np
from RLalgs.utils import epsilon_greedy
from RLalgs.utils import action_evaluation
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

   
    #np.random.seed(6885)
    env.reset()
    q=np.zeros([env.nS,env.nA])
    #q=np.random.normal(0.3,0.1,size=(env.nS,env.nA))
    q[15,:]=0
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        env.reset()
        env.s=np.random.randint(16,size=(1))[0]
        s=env.s
     
        terminal=False
        while terminal!=True:   
            
            a=epsilon_greedy(q[s,:], e)
            out = env.step(a[0])
            q[s,a[0]]=q[s,a[0]]+lr*(out[1] + gamma*q[out[0],np.argmax(q[out[0],:])] - q[s,a[0]])
            s=out[0]
            terminal=out[2]
            
            #env.render()
        #print(i)
    # YOUR CODE ENDS HERE
    ############################

    return q