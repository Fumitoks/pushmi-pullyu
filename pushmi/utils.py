def numpify(t):
    return t.detach().cpu().numpy()

def numpify_list(l):
    return [numpify(t) for t in l]