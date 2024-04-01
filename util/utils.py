import logging
import os

import numpy as np
from torch import nn
import torch


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()

# 保存日志的原来的代码，将其修改
# def init_log(name, level=logging.INFO):
#     if (name, level) in logs:
#         return
#     logs.add((name, level))
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     ch = logging.StreamHandler()
#     ch.setLevel(level)
#     if "SLURM_PROCID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         logger.addFilter(lambda record: rank == 0)
#     else:
#         rank = 0
#     format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
#     formatter = logging.Formatter(format_str)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     return logger

# 自己改了 将日志写入filename文件
def init_log(name, level=logging.INFO, log_path=None, filename=None):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # If filename is given, add a FileHandler to the logger
    if filename is not None:
        filepath = os.path.join(log_path, filename)
        file_handler = logging.FileHandler(filepath, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 加入ti_loss
class TI_Loss(nn.Module):
    """
    References: https://github.com/TopoXLab/TopoInteraction

    The proposed topological interaction (TI) module encodes topological interactions by computing the critical voxels map.
    The critical voxels map contains the locations which induce errors in the topological interactions.
    The TI loss is introduced based on the topological interaction module.
    """

    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        if len(inclusion) != 0:
            for inc in inclusion:
                temp_pair = []
                temp_pair.append(True)  # type inclusion
                temp_pair.append(inc[0])
                temp_pair.append(inc[1])
                self.interaction_list.append(temp_pair)

        if len(exclusion) != 0:
            for exc in exclusion:
                temp_pair = []
                temp_pair.append(False)  # type exclusion
                temp_pair.append(exc[0])
                temp_pair.append(exc[1])
                self.interaction_list.append(temp_pair)
        else:
            raise ValueError(f"Prams 'inclusion' or 'exclusion' must have values")

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0))

    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            # padding='same' needs pytorch >= 1.9, it means the output shape keep the same with input.
            # When kernel size is 3, padding=1 got the same perform.
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding='same')
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding='same')
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, 1, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(), y[:, 0].long()), dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value
