import numpy as np

class Config:
    thresh = 1.0
    tau = 0.7
    gama = 1.0
    T = 3
    alpha = 2.0

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) //self.total_stride + 1 #
        #self.valid_scope = int((self.instance_size - self.exemplar_size) / self.total_stride / 2)#anchor的范围
        self.valid_scope= self.score_size

config = Config()
