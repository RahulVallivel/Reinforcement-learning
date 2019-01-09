import numpy as np
from RLalgs.utils import action_evaluation

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """

    V = np.zeros(env.nS)
    
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    nS = env.nS
    nA = env.nA
    
    delta=1.00000000001e-7
    
    i=0
    numIterations += 1
    while delta>theta:
        temp=V.copy()
        
        #print(temp)
        q=action_evaluation(env, gamma, V)
       
        for s in range(nS):
       
            V[s]= q[s,np.argmax(q[s,:])]
        #print(np.sum(np.absolute(temp-V)))
        delta=np.minimum(delta,np.sum(np.absolute(temp-V)))
    
    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, V, gamma):

    """ 
    Extract the optimal policy given the optimal value-function.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """
    nS = env.nS
    nA = env.nA
    policy = np.zeros(env.nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    q=action_evaluation(env, gamma, V)
    for s in range(nS):
        policy[s]=np.argmax(q[s,:])

    # YOUR CODE ENDS HERE
    ############################

    return policy