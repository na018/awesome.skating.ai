import argparse
import time
from collections import namedtuple

from skatingAI.modules.train_bg import TrainBG
from skatingAI.modules.train_hp import TrainHP
from skatingAI.modules.train_kps import TrainKP
from skatingAI.utils.DsGenerator import DsGenerator, DsPair
from skatingAI.utils.hyper_paramater import HyperParameter
from skatingAI.utils.train_program_menu import TrainProgram
from skatingAI.utils.utils import set_gpus, Logger


class MainLoop(object):
    def __init__(self, name: str, gpu: int, epochs: int, epoch_steps: int, epoch_start: int,
                 epoch_log_n, batch_size: int,
                 params_BG: HyperParameter, params_HP: HyperParameter, params_KP: HyperParameter,
                 w_counter_BG: int, w_counter_HP: int, w_counter_KP: int,
                 do_train_BG: bool, do_train_HP: bool, do_train_KP: bool,
                 ):

        set_gpus(gpu)
        self.epochs, self.epoch_steps, self.epoch_start = epochs, epoch_steps, epoch_start
        self.batch_size, self.epoch_log_n = batch_size, epoch_log_n

        self.do_train_BG, self.do_train_HP, self.do_train_KP = do_train_BG, do_train_HP, do_train_KP
        self.lg = Logger()

        self.generator, self.iter, self.sample_pair = self._generate_dataset()
        _, self.iter_test, _ = self._generate_dataset(test=True)
        self.img_shape = self.sample_pair['frame'].shape

        self.trainBG = TrainBG(params_BG.model, name, self.img_shape,
                               params_BG.optimizer_name, params_BG.learning_rate,
                               params_BG.loss_fct, params_BG.params, params_BG.description,
                               do_train_BG,
                               w_counter_BG, gpu, epochs)

        self.trainHP = TrainHP(params_HP.model, name, self.img_shape,
                               params_HP.optimizer_name, params_HP.learning_rate,
                               params_HP.loss_fct, params_HP.params, params_HP.description,
                               do_train_HP,
                               w_counter_HP, gpu, epochs, self.trainBG.model)

        self.trainKP = TrainKP(params_KP.model, name, self.img_shape,
                               params_KP.optimizer_name, params_KP.learning_rate,
                               params_KP.loss_fct, params_KP.params, params_KP.description,
                               do_train_KP,
                               w_counter_KP, gpu, epochs, self.trainBG.model, self.trainHP.model)

    def _generate_dataset(self, test: bool = False, sequential: bool = False):
        generator = DsGenerator(resize_shape_x=240, test=test, sequential=sequential)
        sample_pair: DsPair = next(generator.get_next_pair())
        ds = generator.build_iterator(self.batch_size, 1)

        return generator, ds.as_numpy_iterator(), sample_pair

    def start_train_loop(self):

        time_start = 0

        if self.epoch_start == -1:
            start = 0
        else:
            start = self.epoch_start

        for epoch in range(start, self.epochs + start):

            if epoch == 3:
                time_start = time.perf_counter()

            for step in range(self.epoch_steps):
                self.lg.log(f'[{step}] train step', True)

                self.trainBG.train_model(self.iter)
                self.trainHP.train_model(self.iter)
                self.trainKP.train_model(self.iter)

            if epoch == start + 3:
                self.trainBG.track_metrics_on_train_start(self.do_train_HP, self.do_train_KP,
                                                          f"{time.perf_counter() - time_start:#.2f}s",
                                                          self.epoch_steps, self.batch_size)
                self.trainHP.track_metrics_on_train_start(self.do_train_BG, self.do_train_KP,
                                                          f"{time.perf_counter() - time_start:#.2f}s",
                                                          self.epoch_steps, self.batch_size)
                self.trainKP.track_metrics_on_train_start(self.do_train_BG, self.do_train_HP,
                                                          f"{time.perf_counter() - time_start:#.2f}s",
                                                          self.epoch_steps, self.batch_size)

            if epoch % self.epoch_log_n == 0:
                self.trainBG.test_model(epoch, self.epoch_steps, self.iter_test)
                self.trainBG.track_logs(self.sample_pair['frame'],
                                        self.sample_pair['mask_bg'],
                                        epoch)
                self.trainHP.test_model(epoch, self.epoch_steps, self.iter_test)
                self.trainHP.track_logs(self.sample_pair['frame'],
                                        self.sample_pair['mask_hp'],
                                        epoch)
                self.trainKP.test_model(epoch, self.epoch_steps, self.iter_test)
                self.trainKP.track_logs(self.sample_pair['frame'],
                                        self.sample_pair['kps'],
                                        epoch)

                self.lg.log(
                    f"[{epoch}:{self.epochs + start}]: Training finished.")

        self.trainBG.progress_tracker.log_on_train_end()


if __name__ == "__main__":
    ArgsNamespace = namedtuple('ArgNamespace',
                               ['gpu', 'name', 'wcounter', 'wcounter_base', 'lr', 'decay', 'opt', 'bs', 'steps',
                                'epochs', 'log_n',
                                'bg'])

    parser = argparse.ArgumentParser(
        description='Train skatingAIs awesome network :)')
    parser.add_argument('--gpu', default=1, help='Which gpu shoud I use?', type=int)
    parser.add_argument('--name', default="kpsdetector_relu_reduce_max_hrnet_v7", help='Name for training')
    parser.add_argument('--wcounter', default=-1, help='Weight counter', type=int)
    parser.add_argument('--wcounter_base', default=4400, help='Weight counter for base net', type=int)
    parser.add_argument('--lr', default=0.005, help='Initial learning rate', type=float)
    parser.add_argument('--decay', default=0.01, help='learning rate decay', type=float)
    parser.add_argument('--opt', default="adam", help='Optimizer [nadam, adam, sgd, sgd_clr]')
    parser.add_argument('--bs', default=3, help='Batch size', type=int)
    parser.add_argument('--steps', default=64, help='Epoch steps', type=int)
    parser.add_argument('--epochs', default=5556, help='Epochs', type=int)
    parser.add_argument('--log_n', default=5, help='Epoch steps', type=int)
    parser.add_argument('--bg', default=True, help='Use training images with background', type=bool)
    args: ArgsNamespace = parser.parse_args()

    optimizer = args.opt
    lr = args.lr
    name = args.name

    general_param, bg_params, hps_params, kps_params, train_bg, train_hp, train_kps = TrainProgram().create_menu()

    if train_bg:
        name = f"bg_{bg_params.name}_"

    if train_hp:
        name += f"hp_{hps_params.name}_"

    if train_kps:
        name += f"bg_{kps_params.name}"

    MainLoop(name=name, gpu=general_param.gpu, epochs=general_param.epochs,
             epoch_steps=general_param.epoch_steps, epoch_start=general_param.epoch_start,
             epoch_log_n=general_param.epoch_log_n, batch_size=general_param.batch_size,
             params_BG=bg_params, params_HP=hps_params, params_KP=kps_params,
             w_counter_BG=general_param.wcounter_bg, w_counter_HP=general_param.wcounter_hp,
             w_counter_KP=general_param.wcounter_kps,
             do_train_BG=train_bg, do_train_HP=train_hp, do_train_KP=train_kps,
             ).start_train_loop()
