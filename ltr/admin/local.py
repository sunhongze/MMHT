class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '.'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.fe108_dir = '/FE108' # This is the path of FE108 dataset
