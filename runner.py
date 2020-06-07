#!/usr/bin/env python
def run(config):
    if config['solver']['record']['iterations']:
        iterations = {algorithm: {} for algorithm in config['solver']['algorithms']}
    if config['solver']['record']['times']:
        times = {algorithm: {} for algorithm in config['solver']['algorithms']}

    if config['problem']['L'] is not None:
        lambda_ = .4 * config['problem']['L']
    else:
        lambda_initial = 1

    for size in config['solver']['sizes']:
        if config['solver']['parameters']['initial'] == 'random':
            x_initial = np.rand.randn(size)
        else:
            x_initial = np.ones(size) * config['solver']['parameters']['initial']

        for algorithm in config['solver']['algorithm']:
            if algorithm == 'korpelevich':
                x, i, t = src.korpelevich()
