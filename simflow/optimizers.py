import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    '''
    Optimizer Class:
    Used to update parameters

    **note**
    this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    '''

    def __init__(self, **kwargs):
        '''
        Initializes updates and weights to empty list

        Args:
            clipvalue (float): value to clip gradients to
        '''
        allowed_kwargs = {'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_var_and_grads(self, vars_and_grads):
        '''
        splits vars_and_grads into variables and gradients
        also clips gradients to clipvalue chosen before

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]): variables and gradients
        Returns:
            params (List[np.ndarray]): pointers to variable to be updated
            grads (List[np.ndarray]): pointers to gradients to be updated
        '''
        try:
            params, grads = zip(*vars_and_grads)
        except ValueError:
            raise ValueError('no gradients found please re-check')

        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [np.clip(g, -self.clipvalue, self.clipvalue)
                     for g in grads]
        return params, grads

    @abstractmethod
    def update_step(self, vars_and_grads):
        '''abstractmethod update_step

        updates vara and grads
        '''


class SGD(Optimizer):
    '''Stochastic Gradient Descent

    Stochastic Gradient Descent combined method of the following methods of
    gradient Descent

    Vanilla:
        Takes a step proportional to the negative of the gradient
            W(t) = W(t-1) - lr * grad

    Momentum:
        Keeps track of velocity and takes a step in the direction of velocity
            v(t) = momentum*v(t-1) - lr*grad
            W(t) = W(t-1) + v(t)

    Nestrov Momentum:
        Smarter momentum step in direction of velocity, correct based on grad
            v(t) = momentum*v(t-1) - lr*grad
            W(t) = W(t-1) -lr*grad + momentum*v(t)
    '''

    def __init__(self, lr=0.01, momentum=0, decay=0, nestrov=False, **kwargs):
        """Setup SGD

        Args:
            lr(float): learning rate [default = 0.01]
            momentum (float): momentum factor used [default = 0]
            decay(float): decay factor by which learning rate
                reduces [default = 0]
            nestrov (bool): set True to enable Nestrov  [default = False]
            clipvalue (float): value to clip gradients to [default = inf]
        """
        super(SGD, self).__init__(**kwargs)
        assert isinstance(lr, float)
        assert isinstance(momentum, (float, int))
        assert isinstance(decay, (float, int))
        assert isinstance(nestrov, bool)
        assert decay >= 0, "-ve decay not valid"
        assert momentum >= 0, "-ve momentum not valid"
        assert momentum < 1, f"momentum should be <1, currently {momentum}"
        assert lr > 0, f"lr should be >0,currently {lr}"
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nestrov = nestrov
        self.iter = 0

    def update_step(self, vars_and_grads):
        '''
        updates vara and grads using SGD

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]) : variables and gradients
        '''
        params, grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self, 'v_grads') and self.momentum:
            # for the first time we need  to init v_grads with zeros
            self.v_grads = [np.zeros_like(g) for g in grads]
        self.iter += 1
        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        if self.momentum:
            for p, g, v in zip(params, grads, self.v_grads):
                v *= self.momentum
                v -= (lr * g)
                if self.nestrov:
                    p += self.momentum*v - lr*g
                else:
                    p += v
        else:
            for p, g in zip(params, grads):
                p -= lr*g


class RMSProp(Optimizer):
    '''RMSProp

    Gradient Descent using RMSProp algorithm

    References:
        - [rmsprop: Divide the gradient by a running average
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    '''

    def __init__(self, lr=0.001, rho=0.9, decay=0, eps=1e-7, **kwargs):
        """Setup RMSProp

        Args:
            lr(float): learning rate [default = 0.001]
            rho (float): RMSProp decay factor used [default = 0.9]
            decay(float): decay factor [default = 0]
            eps (float): Fuzziness factor  [default = 1e-7]
            clipvalue (float): value to clip gradients to [default = inf]
        """
        super(RMSProp, self).__init__(**kwargs)
        assert isinstance(lr, float)
        assert isinstance(rho, (float, int))
        assert isinstance(decay, (float, int))
        assert isinstance(eps, (float, int))

        assert decay >= 0, "-ve decay not valid"
        assert rho >= 0, "-ve rho not valid"
        assert rho < 1, f"rho should be <1, currently {rho}"
        assert lr > 0, f"lr should be >0,currently {lr}"
        self.lr = lr
        self.rho = rho
        self.decay = decay
        self.iter = 0
        self.eps = eps
        # precaution to avoid divide by zero
        if self.eps == 0:
            self.eps = 1e-7

    def update_step(self, vars_and_grads):
        '''
        updates vara and grads using RMSProp

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]): variables and gradients
        '''
        params, grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self, 'a_grads'):
            # for the first time we need  to init a_grads with zeros
            self.a_grads = [np.zeros_like(g) for g in grads]

        self.iter += 1
        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        for p, g, a in zip(params, grads, self.a_grads):
            a *= self.rho
            a += (1-self.rho) * np.square(g)
            p -= lr * (g/(np.sqrt(a)+self.eps))


class Adagrad(Optimizer):
    '''
    Adagrad Gradient Descent

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    Suffers from the inherent problem of early stopping

    References:
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    '''

    def __init__(self, lr=0.01, decay=0, eps=1e-7, **kwargs):
        """Setup Adagrad

        Args:
            lr(float): learning rate [default = 0.01]
            decay(float): decay factor [default = 0]
            eps (float): Fuzziness factor  [default = 1e-7]
            clipvalue (float): value to clip gradients to [default = inf]
        """
        super(Adagrad, self).__init__(**kwargs)
        assert isinstance(lr, float)
        assert isinstance(decay, (float, int))
        assert isinstance(eps, (float, int))

        assert decay >= 0, "-ve decay not valid"
        assert lr > 0, f"lr should be >0,currently {lr}"
        self.lr = lr
        self.decay = decay
        self.iter = 0
        self.eps = eps
        if self.eps == 0:
            self.eps = 1e-7

    def update_step(self, vars_and_grads):
        '''
        updates vara and grads using Adagrad

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]): variables and gradients
        '''
        params, grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self, 'a_grads'):
            # for the first time we need  to init a_grads with zeros
            self.a_grads = [np.zeros_like(g) for g in grads]
        self.iter += 1
        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        for p, g, a in zip(params, grads, self.a_grads):
            a += np.square(g)
            p -= lr * (g/(np.sqrt(a)+self.eps))


class Adadelta(Optimizer):
    '''
    Adadelta Gradient Descent

    Adadelta is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    But overcomes the inherent problem of early stopping in Adadelta,
    since it accumulates only gradients within a fixed window.

    References:
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
    '''

    def __init__(self, lr=1, rho=0.95, decay=0, eps=1e-7, **kwargs):
        """

        Args:
            lr(float): learning rate [default = 0.001]
            rho(float): Adadelta decay factor
            decay(float): decay factor [default = 0]
            eps (float): Fuzziness factor  [default = 1e-7]
            clipvalue (float): value to clip gradients to [default = inf]
        """
        super(Adadelta, self).__init__(**kwargs)
        assert isinstance(lr, (float, int))
        assert isinstance(decay, (float, int))
        assert isinstance(rho, (float, int))
        assert isinstance(eps, (float, int))

        assert decay >= 0, "-ve decay not valid"
        assert lr > 0, f"lr should be > 0,currently {lr}"
        assert rho >= 0, f"rho should be >= 0,currently {rho}"
        assert rho < 1, f"rho should be < 1, currently {rho}"

        self.lr = lr
        self.rho = rho
        self.decay = decay
        self.iter = 0
        self.eps = eps
        if self.eps == 0:
            self.eps = 1e-7

    def update_step(self, vars_and_grads):
        '''
        updates vara and grads using Adagrad

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]) : variables and gradients
        '''
        params, grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self, 'a_grads'):
            # for the first time we need  to init a_grads with zeros
            self.a_grads = [np.zeros_like(g) for g in grads]
        if not hasattr(self, 'da_grads'):
            # for the first time we need  to init da_grads with zeros
            self.da_grads = [np.zeros_like(g) for g in grads]

        self.iter += 1

        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        for p, g, a, d_a in zip(params, grads, self.a_grads, self.da_grads):
            a *= self.rho
            a += (1-self.rho) * np.square(g)

            update = g * np.sqrt(d_a+self.eps) / np.sqrt(a+self.eps)

            p -= lr * update

            d_a *= self.rho
            d_a += (1-self.rho)*np.square(update)


class Adam(Optimizer):
    '''
    Adam
    Adaptive Moment Estimation in short combination of momentum and RMSProp

    References:
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    '''

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, decay=0,
                 eps=0, amsgrad=False, curve_correction=False, **kwargs):
        """Setup Adam

        Args:
            lr(float): learning rate [default = 0.001]
            beta_1 (float): first moment factor used [default = 0.9]
            beta_2 (float): second moment factor used [default = 0.999]
            decay(float): decay rate [default = 0]
            amsgrad (bool): set True to enable AMSGrad  [default = False]
            curve_correction (bool): set to True to enable
                curve_correction of grads [defaul = False]
            clipvalue (float): value to clip gradients to [default = inf]
        """
        super(Adam, self).__init__(**kwargs)
        assert isinstance(lr, float)
        assert isinstance(beta_1, (float, int))
        assert isinstance(beta_2, (float, int))

        assert isinstance(decay, (float, int))
        assert isinstance(amsgrad, bool)
        assert isinstance(curve_correction, bool)

        assert decay >= 0, "-ve decay not valid"
        assert beta_1 >= 0 and beta_2 >= 0, "-ve moments not valid"
        assert beta_1 < 1 and beta_2 < 1, f"both moments should be <1"
        assert lr > 0, f"lr should be >0,currently {lr}"
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.amsgrad = amsgrad
        self.iter = 0
        self.eps = eps
        self.curve_correction = curve_correction
        if self.eps == 0:
            self.eps = 1e-7

    def update_step(self, vars_and_grads):
        '''
        updates vara and grads using SGD

        Args:
            vars_and_grads (List[Tuple[np.ndarray]]) : variables and gradients
        '''
        params, grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self, 'm_grads'):
            # for the first time we need  to init m_grads with zeros
            self.m_grads = [np.zeros_like(g) for g in grads]
        if not hasattr(self, 'v_grads'):
            # for the first time we need  to init v_grads with zeros
            self.v_grads = [np.zeros_like(g) for g in grads]

        if not hasattr(self, 'vhats'):
            if self.amsgrad:
                self.vhats = [np.zeros_like(g) for g in grads]
            else:
                self.vhats = [np.zeros(1) for _ in grads]

        self.iter += 1

        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        for p, g, m, v, vhat in zip(params, grads, self.m_grads,
                                    self.v_grads, self.vhats):
            # update first moment
            m *= self.beta_1
            m += (1-self.beta_1) * g

            # curve correction
            if self.curve_correction:
                m_c = m / (1-self.beta_1**self.iter)
            else:
                m_c = m

            # update second moment
            v *= self.beta_2
            v += (1-self.beta_2) * np.square(g)
            # curve_correction
            if self.curve_correction:
                v_c = v / (1-self.beta_2**self.iter)
            else:
                v_c = v

            if self.amsgrad:
                vhat = np.maximum(vhat, v_c)
                p -= lr * m_c / (np.sqrt(vhat)+self.eps)
            else:
                p -= lr * m_c / (np.sqrt(v_c)+self.eps)
