import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.basic.logger import Writer
from src.utils.basic.io import save_checkpoint
from src.utils.model.optimizer import NoamLR

class ModelBase(nn.Module):

    def __init__(self, config, save_dir=None):
        super(ModelBase, self).__init__()

        self._config = config

        # _config is used to reconstruct the model architecture when reload model from the disk

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.debug = Writer(os.path.join(self.save_dir, "debug.log"))
            self.info  = Writer(os.path.join(self.save_dir, "verbose.log"))
        else:
            self.debug = self.info = print

        self._build(config)  # _build function should only be related with the model structure

        self.epoch_num = 0
        self.batch_num = 0
        self.best_eval_metric = np.inf if config.minimize_score else -np.inf
        self.best_epoch = 0
        self.best_epoch_not_change = 0
        self.stop_flag = False

        self.batch_size        = config.batch_size
        self.device            = config.device
        self.optimizer         = optim.Adam(self.parameters(), lr=config.init_lr)

    def reset_training_records(self):
        self.epoch_num = 0
        self.batch_num = 0
        self.best_eval_metric = np.inf if self._config.minimize_score else -np.inf
        self.best_epoch = 0
        self.best_epoch_not_change = 0
        self.stop_flag = False

    def save_model(self, path):
        save_checkpoint(path=path,
                        model=self,
                        model_type=self._model_type,
                        args=self._config)

    def reset_save_dir(self, save_dir):

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.debug = Writer(os.path.join(self.save_dir, "debug.log"))
            self.info  = Writer(os.path.join(self.save_dir, "verbose.log"))
        else:
            self.debug = self.info = print

    @property
    def _model_type(self):
        raise NotImplementedError

    def _build(self, config):
        raise NotImplementedError

    def _forward(self, inputs):
        """
        :param inputs: the first parameter returned by collate_fn
        :return: outputs
        """
        raise NotImplementedError

    def _loss(self,
              outputs,
              targets):
        raise NotImplementedError

    def predict_on_dataloader(self, dataloader: DataLoader):
        raise NotImplementedError

    def eval_on_dataloader(self, dataloader: DataLoader):
        raise NotImplementedError

    def report_on_dataloader(self, dataloader, path):
        raise NotImplementedError

    def forward(self, inputs):
        """Wrapper for _forward()"""
        self.train()
        return self._forward(inputs)

    def predict_on_batch(self, inputs):
        self.eval()
        with torch.no_grad():
            return self._forward(inputs)

    def fit_on_batch(self,
                     batch_inputs,
                     batch_targets):
        """
        Train the model for given inputs
        inputs: batch_inputs
        targets: batch_targets
        """
        outputs = self.forward(batch_inputs)
        loss = self._loss(outputs, batch_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit_on_dataloader(self,
                          train_dataloader: DataLoader,
                          eval_dataloader: DataLoader):
        """
        Train the model for a whole dataloader
        :return:
        """
        if os.path.exists(os.path.join(self.save_dir, "trained_model.config")):
            raise IOError("The directory for saving model is not empty.")

        self.reset_training_records()

        while True:
            self.epoch_num += 1
#            self.batch_num = 0
            for batch in train_dataloader:
                self.batch_num += 1
                inputs  = batch[0]
                targets = batch[1]
                self.fit_on_batch(inputs, targets)

            # check if need epoch-based evaluation

                if self.batch_num % self._config.log_frequency == 0 and self.epoch_num >= self._config.at_least_epoch:
                # if self.batch_num % self._config.log_frequency == 0:
                    # self.eval_on_dataloader(eval_dataloader)
                # if self.batch_num % self._config.log_frequency == 0 and self.epoch_num >= 10:
                    self._check_stopping(eval_dataloader)
                    if self.stop_flag is True:
                        self.debug('Finish Training.')
                        self.debug(f'Best {self._config.metric}: {self.best_eval_metric:.3f}')
                        return self.best_eval_metric

    def _check_stopping(self, eval_dataloader):
        self.debug("")
        self.debug(f"Start Evaluating on Evaluation Set, EPOCH {self.epoch_num} BATCH {self.batch_num}")
        metrics = self.eval_on_dataloader(eval_dataloader)
        for metric in metrics:
            self.debug(f"{metric}: {np.mean(metrics[metric]):.3f}")
        indicator = np.mean(metrics[self._config.metric])

        # Use the main metrics (the first one if multiple metrics are provided) to check if stop
        if (indicator > self.best_eval_metric and self._config.minimize_score == False) or (indicator < self.best_eval_metric and self._config.minimize_score):
            self.best_eval_metric = indicator
            self.best_eval_epoch  = self.epoch_num
            self.best_epoch_not_change   = 0
            self.save_model(path=os.path.join(self.save_dir, "trained_model.config"))
            return None
        else:
            self.best_epoch_not_change  += 1
            self.debug(f"Model not improve for {self.best_epoch_not_change } Epochs.")
            self.debug(f"Best {self._config.metric} Now: {self.best_eval_metric:.3f}")
            if self.best_epoch_not_change >= self._config.early_stopping_num:
                self.stop_flag = True
            return None