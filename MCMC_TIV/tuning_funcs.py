'''
Checks acceptance rate for this parameter (need to input param-specific values and modifies w accordingly
    If too low (<0.3) multiply parameter's w by 0.75
    If too high (>0.5), multiply w by 1.5
Returns w
'''

def tune_w(rejects, current_iter, w):
    acceptance_rate = 1 - rejects/current_iter

    if acceptance_rate < 0.3:
        w *= 0.75

    if acceptance_rate >0.5:
        w *= 1.5

    return w