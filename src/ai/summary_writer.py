"""Drop-in replacement for tensorboard SummaryWriter with some fixes."""

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class CustomSummaryWriter(SummaryWriter):
    def add_hparams(
        self,
        hparam_dict,
        metric_dict,
        hparam_domain_discrete=None,
        global_step=None,
    ):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        file_writer = self._get_file_writer()
        file_writer.add_summary(exp, global_step)
        file_writer.add_summary(ssi, global_step)
        file_writer.add_summary(sei, global_step)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)
