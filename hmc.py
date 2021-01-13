import torch

def hamiltonian(x, v, model):
    energy = 0.5 * torch.pow(v, 2).sum(dim=1) + model(x).squeeze()
    return energy


def leapfrog_step(x, v, model, step_size, num_steps):
    # x = torch.log(x / (1 - x + 1e-10))
    x.requires_grad_(requires_grad=True)
    #energy = model(torch.sigmoid(x))
    energy = model(x)
    im_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * step_size * im_grad


    for _ in range(num_steps):
        x.requires_grad_(requires_grad=True)
        energy = model(x)

        im_grad = torch.autograd.grad([energy.sum()], [x])[0]
        v = v - step_size * im_grad
        x = x + step_size * v
        x = x.detach()
        v = v.detach()

  
    x.requires_grad_(requires_grad=True)
    #energy = model(torch.sigmoid(x))
    energy = model(x)
    im_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * im_grad
    x = x.detach()
    #x = torch.sigmoid(x.detach())

    return x, v, im_grad


def sample_hmc(x, model_fn, step_size, num_steps=10):
    v = 0.1 * torch.randn_like(x)

    x_new, v_new, grad = leapfrog_step(x, v, model_fn, step_size, num_steps)

    orig = hamiltonian(x, v, model_fn)
    new = hamiltonian(x, v_new, model_fn)

    mask = (torch.exp((orig - new)) > (torch.rand(new.size(0))).to(x.device))
    x_new[mask]= x[mask]
    return x_new
