from keras.callbacks import Callback
from keras import backend as K

import math


class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_max, eta_min=0):
        """Constructor

        @param T_max:               number of epochs needed to go from eta_max to eta_min
        @param eta_max:             max (and start) value of learning rate
        @param eta_min:             min value of learning rate
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min

    def on_epoch_begin(self, epoch, logs=None):
        """Method called on epoch start

        @param epoch:               current epoch
        @type logs:                 logs for tensorboard
        """
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        """Method called on epoch end

        @param epoch:               current epoch
        @type logs:                 logs for tensorboard
        """
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
