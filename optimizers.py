import numpy as np
class Optimizer(object):
    '''
    Optimizer Class:
    Used to update parameters
    '''
    def __init__(self, **kwargs):
        '''
        Initializes updates and weights to empty list

        Args:
            clipvalue (float) : value to clip gradients to
        '''
        allowed_kwargs = {'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_var_and_grads(self,vars_and_grads):
        '''
        splits vars_and_grads into variables and gradients
        also clips gradients to clipvalue chosen before

        Args:
            vars_and_grads (list of tuples of numpy.ndarray) : list of tuples of variable and gradient to be updated
        Returns:
            params (list of numpy.ndarray) : pointers to variable to be updated
            grads (list of numpy.ndarray)  : pointers to gradients to be updated
        '''
        try:
            params,grads = zip(*vars_and_grads)
        except ValueError:
            raise ValueError('no gradients found please reacheck')
        # if None in gs:
        #     raise ValueError('One of your gradients have an undefined gradient please check')
        if hasattr(self,'clipvalue') and self.clipvalue>0:
            grads = [np.clip(g,-self.clipvalue,self.clipvalue) for g in grads]
        return params,grads

    def update_step(self,vars_and_grads):
        '''
        updates vara and grads
        '''
        raise NotImplementedError('optimizer not defined')



class SGD(Optimizer):
    '''
    Stochastic Gradient Descent

    Args:
        lr(float)           :   learning rate [default = 0.01]
        momentum (float)    :   momentum factor used [default = 0]
        decay(float)        :   decay factor by which learning rate reduces [default = 0]
        nestrov (bool)      :   set True to enable Nestrov  [default = False]
        clipvalue (float)   :   value to clip gradients to [default = inf]
    '''
    def __init__(self,lr=0.01,momentum=0,decay=0,nestrov=False,**kwargs):
        super(SGD,self).__init__(**kwargs)
        assert decay >= 0,"-ve decay not valid"
        assert momentum >=0,"-ve momentum not valid"
        assert momentum <1,f"momentum should be <1, currently {momentum}"
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nestrov = nestrov
        self.iter = 0

    def update_step(self,vars_and_grads):
        '''
        updates vara and grads using SGD
        Args:
            vars_and_grads (list of tuples of numpy.ndarray) : list of tuples of variable and gradient to be updated
        '''
        params,grads = self.get_var_and_grads(vars_and_grads)
        if not hasattr(self,'m_grads') and self.momentum:
            #for the first time we need  to init m_grads with zeros
            self.m_grads = [np.zeros_like(g) for g in grads]
        self.iter+=1
        lr = self.lr
        lr *= (1/(1+self.decay*self.iter))

        if self.momentum:
            for p,g,m in zip(params,grads,self.m_grads):
                m *= self.momentum
                m -= (lr * g)
                if self.nestrov:
                    p += self.momentum*m - lr*g
                else:
                    p += m
        else:
            for p,g in zip(params,grads):
                p -= lr*g
