# from https://arxiv.org/pdf/2103.13620.pdf

def SubspectralNorm(x, gamma, beta, S, eps=1e-5):
    N, C, F, T = x.size()
    x = x.view(N, C*S, F//S, T)

    mean = x.mean([0, 2, 3]).view([1, C*S, 1, 1])
    var = x.var([0, 2, 3]).view([1, C*S, 1, 1])
    x = gamma * (x - mean) / (var + eps).sqrt() + beta

    return x.view(N, C, F, T)
