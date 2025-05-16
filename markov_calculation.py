import numpy as np
from collections import defaultdict

def build_matrices(transitions, states, actions, absorbing_states):
    state_action_counts = defaultdict(lambda: defaultdict(int))
    state_action_rewards = defaultdict(lambda: defaultdict(float))
    state_action_next = defaultdict(lambda: defaultdict(int))

    for (s, a, r), (s_next, _, _) in zip(transitions, transitions[1:]):
        if s in absorbing_states:
            continue  
        state_action_counts[s][a] += 1
        state_action_rewards[s][a] += r
        state_action_next[(s, a)][s_next] += 1

    P = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for (s, a), next_states in state_action_next.items():
        total_transitions = sum(next_states.values())
        for s_next, count in next_states.items():
            P[s][a][s_next] = count / total_transitions

    for s in absorbing_states:
        P[s] = {a: {s: 1.0} for a in actions}  
        state_action_rewards[s] = {a: 0 for a in actions}  

    R = defaultdict(lambda: defaultdict(float))
    for s, actions in state_action_counts.items():  
        for a, count in actions.items():
            R[s][a] = (
                state_action_rewards[s][a] / count
                if count > 0 else 0
            )

    for s in states:
        if s not in P:
            P[s] = {a: {s: 0.0} for a in actions}  
        if s not in R:
            R[s] = {a: 0 for a in actions}  
        for a in actions:
            if a not in P[s]:
                P[s][a] = {s: 0.0}
            if a not in R[s]:
                R[s][a] = 0 

    return P, R

def solve_bellman(P, R, states, actions, absorbing_states, gamma=0.9, epsilon=1e-6):
    V = {s: 0 if s not in absorbing_states else max(R.get(s, {}).values(), default=0) for s in states}

    while True:
        delta = 0
        for s in states:
            if s in absorbing_states:
                continue
            
            v = V[s]
            V[s] = max(
                sum(P[s][a].get(s_next, 0) * (R[s].get(a, 0) + gamma * V.get(s_next, 0))
                    for s_next in states)
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < epsilon:
            break

    return V

def compute_optimal_policy(P, R, states, actions, V, gamma):
    policy = {}
    for s in states:
        policy[s] = max(actions, key=lambda a: 
            sum(P[s][a].get(s_next, 0) * (R[s].get(a, 0) + gamma * V.get(s_next, 0))
                for s_next in states))
    return policy

def print_state_transition_matrix(P, states):
    print("State transition probability matrix:")
    matrix = np.zeros((len(states), len(states)))
    
    for i, s in enumerate(states):
        total_probability = 0
        for j, s_next in enumerate(states):
            prob = sum(P[s][a].get(s_next, 0) for a in P[s])
            matrix[i, j] = prob
            total_probability += prob
        
        if total_probability > 0:
            matrix[i, :] /= total_probability
    
    print(matrix)


def main(data_tuple, states, actions, absorbing_states, gamma=0.9):
    transitions = [(data_tuple[i], data_tuple[i+1], data_tuple[i+2]) for i in range(0, len(data_tuple), 3)]

    P, R = build_matrices(transitions, states, actions, absorbing_states)
    
    print_state_transition_matrix(P, states)
    
    V = solve_bellman(P, R, states, actions, absorbing_states, gamma)
    
    optimal_policy = compute_optimal_policy(P, R, states, actions, V, gamma)
    
    return V, optimal_policy


data_tuple = [1,1,1,1,2,3,1,2,1,1,2,1,2,5,1,2,5,1,2,5,3,3,5,1,3,5,-3,1,3,1,1,3,3,3,5,3,3,6,3,4,6,3,1,1,1,1,2,1,1,1,3,2,5,1,2,5,1,2,5,3,3,5,1,3,5,1,3,5,2,3,6,3,4,6,3,1,1,1,1,3,3,2,2,1,2,5,3,3,5,1,3,5,-3,1,2,0,1,2,0,1,3,1,1,3,3,3,5,1,3,5,1,3,5,1,3,5,-3,1,3,1,1,3,1,1,3,3,3,3,1,3,3,1,3,6,3,4,6,1]
states = [1, 2, 3, 4, 5]  
actions = [1, 2, 3, 4, 5, 6]  
absorbing_states = [4, 5]
gamma = 0.9

V, optimal_policy = main(data_tuple, states, actions, absorbing_states, gamma)

print("Value function:", V)
print("Optimal policy:", optimal_policy)
