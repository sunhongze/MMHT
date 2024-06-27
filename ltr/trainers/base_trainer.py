import os
import glob
import torch
import traceback
from ltr.admin import loading, multigpu


class BaseTrainer:
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def update_settings(self, settings=None):
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True, latest_model=None):
        for epoch in range(self.epoch+1, max_epochs+1):
            self.epoch = epoch
            self.train_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self._checkpoint_dir:
                self.save_checkpoint()
        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        os.rename(tmp_file_path, file_path)
