from typing import List

import tensorflow as tf

import skatingAI.nets.hrnet as hrnet
import skatingAI.nets.keypoint as kp_net
from skatingAI.utils.losses import CILoss


class HyperParameterParams(object):
    def __init__(self, epsilon: float = 1e-8,
                 amsgrad: bool = True,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 decay: float = 0.01,
                 sgd_clr_decay_rate: List[float] = [],
                 epoch_sgd_plateau_check: int = 50):
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.sgd_clr_decay_rate = sgd_clr_decay_rate
        self.epoch_sgd_plateau_check = epoch_sgd_plateau_check


class GeneralTrainParams(object):
    def __init__(self, gpu: int = 1,
                 wcounter_hp: int = -1,
                 wcounter_kps: int = -1,
                 epoch_start: int = 0,
                 batch_size: int = 3,
                 epoch_steps: int = 64,
                 epochs: int = 5556,
                 epoch_log_n: int = 5):
        self.gpu = gpu
        self.wcounter_hp = wcounter_hp
        self.wcounter_kps = wcounter_kps
        self.epoch_start = epoch_start
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.epochs = epochs
        self.epoch_log_n = epoch_log_n


class HyperParameter(object):
    def __init__(self, name: str, model,
                 optimizer_name: str, learning_rate: float, description: str, loss_fct: tf.keras.losses,
                 params: HyperParameterParams):
        self.name = name
        self.model = model
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.description = description
        self.loss_fct = loss_fct
        self.params = params


BodyPoseDetectorHyperParameters = [
    HyperParameter(
        name='ciloss_adam',
        model=hrnet.v7.HRNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        description='',
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True)
    ),
    HyperParameter(
        name='ciloss_nadam',
        model=hrnet.v7.HRNet,
        optimizer_name='nadam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        description='',
        params=HyperParameterParams(beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    ),
    HyperParameter(
        name='ciloss_sgd',
        model=hrnet.v7.HRNet,
        optimizer_name='sgd',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        description='',
        params=HyperParameterParams()
    ),
    HyperParameter(
        name='ciloss_sgd_decay',
        model=hrnet.v7.HRNet,
        optimizer_name='sgd',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        description='',
        params=HyperParameterParams(decay=0.01)
    ),
    HyperParameter(
        name='ciloss_sgd_clr',
        model=hrnet.v7.HRNet,
        optimizer_name='sgd',
        learning_rate=0.1,
        loss_fct=CILoss(9),
        description='',
        params=HyperParameterParams(sgd_clr_decay_rate=[1e-2, 1e-3, 1e-4, 1e-5], decay=0.1)
    ),
]
KeyPointDetectorHyperParameters = [
    HyperParameter(
        name='conv_relu_sigmoid',
        model=kp_net.v0.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        description='',
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True)
    ),
    HyperParameter(
        name='relu_sigmoid_adam1e3',
        model=kp_net.v1.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description=''
    ),
    HyperParameter(
        name='extended_conv_linear_adam1e3',
        model=kp_net.v2.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description=''
    )
]
