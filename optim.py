def sgd(x, sx, psi_hat, psi, p):
    # collect parameters
    nw = psi_hat.shape[1]
    niter = p.niter
    nbatch = p.nbatch
    eta = p.eta
    
    x0 = x
    for i in range(niter):
        ind = np.random.choice(nw, nbatch)
        j = jacfun(x0, sx[ind], psi_hat[:,ind], psi[:,ind])
        j = j / np.linalg.norm(j, 1) # normalize gradients
        x0 = x0 - eta*j
    return x0

def sgd_with_mom(x, sx, psi_hat, psi, p):
    # collect parameters
    nw = psi_hat.shape[1]
    niter = p.niter
    nbatch = p.nbatch
    eta = p.eta
    gamma = p.gamma
    
    x0 = x
    v0 = 0
    for i in range(niter):
        ind = np.random.choice(nw, nbatch)
        j = jacfun(x0, sx[ind], psi_hat[:,ind], psi[:,ind])
        j = j / np.linalg.norm(j, 1) # normalize gradients
        v = gamma * v0 + eta * j
        x0 = x0 - v
        v0 = v
    return x0

def nag(x, sx, psi_hat, psi, p):
    # collect parameters
    nw = psi_hat.shape[1]
    niter = p.niter
    nbatch = p.nbatch
    eta = p.eta
    gamma = p.gamma
    
    x0 = x
    v0 = 0
    for i in range(niter):
        ind = np.random.choice(nw, nbatch)
        j = jacfun(x0 - gamma * v0, sx[ind], psi_hat[:,ind], psi[:,ind])
        j = j / np.linalg.norm(j, 1) # normalize gradients
        v = gamma * v0 + eta * j
        x0 = x0 - v
        v0 = v
    return x0

def rmsprop(x, sx, psi_hat, psi, p):
    # collect parameters
    nw = psi_hat.shape[1]
    niter = p.niter
    nbatch = p.nbatch
    eta = p.eta
    beta = p.beta
    epsilon = p.epsilon
    
    x0 = x
    s = 0
    for i in range(niter):
        ind = np.random.choice(nw, nbatch)
        j = jacfun(x0, sx[ind], psi_hat[:,ind], psi[:,ind])
        j = j / np.linalg.norm(j, 1) # normalize gradients
        s = beta * s + (1 - beta) * j**2
        x0 = x0 - eta * j / np.sqrt(s + epsilon)
    return x0

def adam(x, sx, psi_hat, psi, p):
    # collect parameters
    nw = psi_hat.shape[1]
    niter = p.niter
    nbatch = p.nbatch
    eta = p.eta
    gamma = p.gamma
    beta = p.beta
    epsilon = p.epsilon
    
    x0 = x
    s = 0
    m = 0
    for i in range(niter):
        ind = np.random.choice(nw, nbatch)
        j = jacfun(x0, sx[ind], psi_hat[:,ind], psi[:,ind])
        j = j / np.linalg.norm(j, 1) # normalize gradients
        m = gamma * m + (1 - gamma) * j
        m = m / (1 - gamma)
        s = beta * s + (1 - beta) * j**2
        s = s / (1 - beta)
        x0 = x0 - eta * m / np.sqrt(s + epsilon)
    return x0
