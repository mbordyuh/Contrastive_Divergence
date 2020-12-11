import numpy as np
import torch
import torch.autograd as autograd


def sample_langevin(x, model, stepsize, n_steps):
    "Langevin sampling"

    noise_scale = np.sqrt(2 * stepsize)
    samples = []
    x.requires_grad = True

    for _ in range(n_steps):
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        noise = torch.randn_like(x) * noise_scale
        x = x + stepsize * grad + noise
        samples.append(x.detach())

    return samples[-1]

