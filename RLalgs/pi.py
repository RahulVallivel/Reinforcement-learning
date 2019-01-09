import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
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
    """

    V = np.zeros(env.nS)
    nS = env.nS
    nA = env.nA
    policy = np.ones(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        numIterations += 1
        
        V=policy_evaluation(env, policy, gamma, theta,V)
        
        policy, policy_stable=policy_improvement(env, V, policy, gamma)
        
        # YOUR CODE ENDS HERE
        ############################
        
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta,V):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################ 
    # YOUR CODE STARTS HERE
    
    nS = env.nS
    nA = env.nA
    
    delta=1.00000000001e-7
    
    i=0
    while delta>theta:
        temp=V.copy()
        
        
        q=action_evaluation(env, gamma, V)
       
        for s in range(nS):
       
            V[s]= q[s,policy[s]]
  
        delta=np.minimum(delta,np.sum(np.absolute(temp-V)))
        
        i=i+1
       

    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    
    nS = env.nS
    nA = env.nA
    q=action_evaluation(env, gamma, value_from_policy)
  
    temp=policy.copy()
    temp1=temp.tolist()
    for s in range(nS):
        policy[s]=np.argmax(q[s,:])
        policy1=policy.tolist()
    if temp1 == policy1:
        policy_stable=True
        return policy, policy_stable
        
    else:
        policy_stable=False
        policy_improvement(env, value_from_policy, policy, gamma)
    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable