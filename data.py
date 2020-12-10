import math
import torch


def sample_data(n_samples):
    '''taken from https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py'''

    z = torch.randn(n_samples, 2)

    scale = 4
    sq2 = 1 / math.sqrt(2)
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2),
               (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]

    centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
    return sq2 * \
        (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])
