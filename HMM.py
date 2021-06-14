import numpy as np

hidden_state = ['sunny', 'rainy']

obsevition = ['walk', 'shop', 'clean']


def calculate(obs, states, start_p, trans_p, emit_p):
    max_p = np.zeros((len(obs), len(states)))

    path = np.zeros((len(states), len(obs)))

    for i in range(len(states)):
        max_p[0][i] = start_p[i] * emit_p[i][obs[0]]
        path[i][0] = i

    for t in range(1, len(obs)):
        new_path = np.zeros((len(states), len(obs)))
        for y in range(len(states)):
            prob = -1
            for y0 in range(len(states)):
                nprob = max_p[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]
                if nprob > prob:
                    prob = nprob
                    state = y0
                    #   记录路径
                    max_p[t][y] = prob
                    for m in range(t):
                        new_path[y][m] = path[state][m]
                    new_path[y][t] = y

        path = new_path

    max_prob = -1
    path_state = 0

    for y in range(len(states)):
        if max_p[len(obs) - 1][y] > max_prob:
            max_prob = max_p[len(obs) - 1][y]
            path_state = y

    return path[path_state]


state_s = [0, 1]
obser = [0, 1, 2]

start_probability = [0.6, 0.4]

transititon_probability = np.array([[0.7, 0.3], [0.4, 0.6]])

emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

result = calculate(obser, state_s, start_probability, transititon_probability, emission_probability)

for k in range(len(result)):
    print(hidden_state[int(result[k])])
