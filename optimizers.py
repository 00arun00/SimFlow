import numpy as np
class Optimizer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_var_and_grads(self,vars_and_grads):
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
        raise NotImplementedError('optimizer not defined')



class SGD(Optimizer):
    def __init__(self,lr=0.01,momentum=0,decay=0,nestrov=False,**kwargs):
        super(SGD,self).__init__(**kwargs)
        assert momentum>=0,"-ve momentum not valid"
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nestrov = nestrov
        self.iter = 0

    def update_step(self,vars_and_grads):
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
