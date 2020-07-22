from typing import List

import tensorflow as tf

import skatingAI.nets.bg as bgnet
import skatingAI.nets.hrnet as hrnet
import skatingAI.nets.keypoint as kp_net
from skatingAI.utils.BGLoss import BGLoss
from skatingAI.utils.KPSLoss import KPSLoss
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
                 wcounter_bg: int = -1,
                 wcounter_hp: int = -1,
                 wcounter_kps: int = -1,
                 epoch_start: int = 0,
                 batch_size: int = 3,
                 epoch_steps: int = 64,
                 epochs: int = 5556,
                 epoch_log_n: int = 5):
        self.gpu = gpu
        self.wcounter_bg = wcounter_bg
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


BGExtractorHyperParameters = [
    HyperParameter(
        name='adam',
        model=bgnet.v7.BGNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=BGLoss(2),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Test **adam** optimizer on hrnet v7',
    ),
    HyperParameter(
        name='adam',
        model=bgnet.v0.BGNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=BGLoss(2),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Test **adam** optimizer on hrnet v0',
    )
]
BodyPoseDetectorHyperParameters = [
    HyperParameter(
        name='chrnet_adam',
        model=hrnet.v7.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Test **adam** optimizer on hrnet v7',
    ),
    HyperParameter(
        name='chrnet_nadam',
        model=hrnet.v7.HPNet,
        optimizer_name='nadam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        description='Test **nadam** optimizer on hrnet v7',

    ),
    HyperParameter(
        name='chrnet_sgd',
        model=hrnet.v7.HPNet,
        optimizer_name='sgd',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(),
        description='Test **sgd** optimizer on hrnet v7',
    ),
    HyperParameter(
        name='chrnet_sgd_decay',
        model=hrnet.v7.HPNet,
        optimizer_name='sgd',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(decay=0.01),
        description='Test **sgd** optimizer <br>'
                    'reduce lr with static decay<br>'
                    'on hrnet v7',
    ),
    HyperParameter(
        name='chrnet_sgd_clr',
        model=hrnet.v7.HPNet,
        optimizer_name='sgd_clr',
        learning_rate=0.1,
        loss_fct=CILoss(9),
        params=HyperParameterParams(sgd_clr_decay_rate=[1e-2, 1e-3, 1e-4, 1e-5], decay=0.1),
        description='Test **sgd** optimizer <br>'
                    'reduce lr if training stocks<br>'
                    'on hrnet v7',
    ),
    HyperParameter(
        name='chrnet_crossentropy_adam',
        model=hrnet.v7.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.SparseCategoricalCrossentropy(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Test cross entropy loss <br>on hrnet v7',
    ),
    HyperParameter(
        name='traditional_hrnet_64',
        model=hrnet.v0.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train HPNet as described in paper',
    ),
    HyperParameter(
        name='traditional_hrnet_adjusted_filter',
        model=hrnet.v1.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Test hrnet with adjusted filters<br> Smaller conv layers learn more coarse filter',
    ),
    HyperParameter(
        name='hrnet_stride_down_up',
        model=hrnet.v2.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Reduce conv size by <br>striding down the input layer',
    ),
    HyperParameter(
        name='hrnet_stride_down_up_input',
        model=hrnet.v3.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Reduce conv size by <br>striding down the input layer<br>Use input layer for every block module.',
    ),
    HyperParameter(
        name='hrnet_3_blocks',
        model=hrnet.v4.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Reduce to 3 blocks, [l,m,s].',
    ),
    HyperParameter(
        name='UNet',
        model=hrnet.v5.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train UNet',
    ),
    HyperParameter(
        name='chrnet_add_dwonv',
        model=hrnet.v6.HPNet,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=CILoss(9),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Add Input layer to all combination layers,<br>'
                    'replace concat with add layers,<br>'
                    'experiment with depthwise convolution',
    ),
]
KeyPointDetectorHyperParameters = [
    HyperParameter(
        name='block_l_relu_sigmoid',
        model=kp_net.v0.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train hrnet block_l <br>'
                    'with 1 MaxPool reducing image size<br>'
                    '& 2 relu+sigmoid Dense modules',
    ),
    HyperParameter(
        name='relu_sigmoid',
        model=kp_net.v1.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train 2 relu+sigmoid Dense modules <br>'
                    'with 1 MaxPool reducing image size'
    ),
    HyperParameter(
        name='extended_conv_relu_sigmoid',
        model=kp_net.v2.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=tf.keras.losses.MeanSquaredError(),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train 4 MaxPool-Conv modules<br>'
                    '& 2 relu+sigmoid Dense modules'
    ),
    HyperParameter(
        name='blockl_kps_loss',
        model=kp_net.v3.KPDetector,
        optimizer_name='adam',
        learning_rate=1e-3,
        loss_fct=KPSLoss(11),
        params=HyperParameterParams(epsilon=1e-8, amsgrad=True),
        description='Train block_l with 11 keypoint classes<br>'
                    '& one background class with KPSLoss'
    ),
    HyperParameter(
        name='blockl_kps_loss_sgd_clr',
        model=kp_net.v3.KPDetector,
        optimizer_name='sgd_clr',
        learning_rate=1e-3,
        loss_fct=KPSLoss(11),
        params=HyperParameterParams(sgd_clr_decay_rate=[1e-2, 1e-3, 1e-4, 1e-5], decay=0.1),
        description='Train block_l with 11 keypoint classes<br>'
                    '& one background class with KPSLoss<br>'
                    '& sgd_clr'
    )
]
