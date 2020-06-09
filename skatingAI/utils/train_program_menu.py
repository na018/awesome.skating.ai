from pick import pick

from skatingAI.utils.hyper_paramater import KeyPointDetectorHyperParameters, BodyPoseDetectorHyperParameters, \
    GeneralTrainParams, BGExtractorHyperParameters


class TrainProgram(object):
    def _parse_if_int(self, value):
        try:
            if int(value) >= 0:
                return True
            else:
                return False
        except:
            return False

    def create_menu(self) -> (
            GeneralTrainParams, BodyPoseDetectorHyperParameters, KeyPointDetectorHyperParameters, bool, bool):

        bg_param: BodyPoseDetectorHyperParameters = BGExtractorHyperParameters[0]
        hp_param: BodyPoseDetectorHyperParameters = BodyPoseDetectorHyperParameters[0]
        kps_param: KeyPointDetectorHyperParameters = KeyPointDetectorHyperParameters[0]
        general_train_params = GeneralTrainParams(
            gpu=1,
            wcounter_bg=-1,
            wcounter_hp=-1,
            wcounter_kps=-1,
            epoch_start=0,
            batch_size=12,
            epoch_steps=64,
            epochs=5556,
            epoch_log_n=5
        )
        train_bg, train_hp, train_kps = False, False, False
        print('\n' * 2)
        print('Welcome to the custom training of body parts and key points of skatingAI ⛸💡')

        title = f'Welcome to the custom training of body parts and key points of skatingAI ⛸💡\n\n' \
                f'Do you want to stick to the default parameters for training?'
        options = [f'yes: {general_train_params.__dict__}', 'No, I want to choose my own.']
        option, index = pick(options, title)

        if index == 0:
            print('Cool, you will stick to the default general setting.')
        else:
            gpu = input(f'Please choose a gpu to train on [0,1,2,3]: [{general_train_params.gpu}] ')
            general_train_params.gpu = int(gpu) if self._parse_if_int(gpu) else general_train_params.gpu
            wcounter_bg = input(
                f'Please choose a weight counter index for the background extraction pre-training: [{general_train_params.wcounter_bg}] ')
            general_train_params.wcounter_bg = int(wcounter_bg) if self._parse_if_int(
                wcounter_bg) else general_train_params.wcounter_bg
            wcounter_hp = input(
                f'Please choose a weight counter index for the body parts pre-training: [{general_train_params.wcounter_hp}] ')
            general_train_params.wcounter_hp = int(wcounter_hp) if self._parse_if_int(
                wcounter_hp) else general_train_params.wcounter_hp
            wcounter_kps = input(
                f'Please choose a weight counter index for the keypoints pre-training: [{general_train_params.wcounter_kps}] ')
            general_train_params.wcounter_kps = int(wcounter_kps) if self._parse_if_int(
                wcounter_kps) else general_train_params.wcounter_kps
            epoch_start = input(
                f'Please choose the epoch you want to start the training from: [{general_train_params.epoch_start}] ')
            general_train_params.epoch_start = int(epoch_start) if self._parse_if_int(
                epoch_start) else general_train_params.epoch_start
            batch_size = input(f'Please choose the batch_size: [{general_train_params.batch_size}] ')
            general_train_params.batch_size = int(batch_size) if self._parse_if_int(
                batch_size) else general_train_params.batch_size
            epochs = input(f'Please choose the maximum epochs for the training loop: [{general_train_params.epochs}] ')
            general_train_params.epochs = int(epochs) if self._parse_if_int(epochs) else general_train_params.epochs
            epoch_steps = input(
                f'Please choose how many steps you want to train for every epoch: [{general_train_params.epoch_steps}] ')
            general_train_params.epoch_steps = int(epoch_steps) if self._parse_if_int(
                epoch_steps) else general_train_params.epoch_steps
            epoch_log_n = input(
                f'Please choose after how many epochs you need logging information: [{general_train_params.epoch_log_n}] ')
            general_train_params.epoch_log_n = int(epoch_log_n) if self._parse_if_int(
                epoch_log_n) else general_train_params.epoch_log_n

        title = f'Which model do you want to train?'
        options = ['background extraction', 'body part detection', 'key point detection', 'background & body part',
                   'all together']
        option, index = pick(options, title)

        if option == 'background extraction' or option == 'background & body part' or option == 'all together':
            title = 'Please choose one of the Hyper Parameter training settings for the background extraction training: '
            options = [f"{parameter.name}: {parameter.description}" for parameter in BGExtractorHyperParameters]
            bg_param_option, index = pick(options, title)
            bg_param = BGExtractorHyperParameters[index]
            train_bg = True

        if option == 'body part detection' or option == 'background & body part' or option == 'all together':
            title = 'Please choose one of the Hyper Parameter training settings for the body part detection training: '
            options = [f"{parameter.name}: {parameter.description}" for parameter in BodyPoseDetectorHyperParameters]
            hp_param_option, index = pick(options, title)
            hp_param = BodyPoseDetectorHyperParameters[index]
            train_hp = True

        if option == 'key point detection' or option == 'all together':
            title = 'Please choose one of the Hyper Parameter training settings for the keypoint detection training:: '
            options = [f"{parameter.name}: {parameter.description}" for parameter in KeyPointDetectorHyperParameters]
            option, index = pick(options, title)
            kps_param = KeyPointDetectorHyperParameters[index]
            train_kps = True

        print('\n' * 2)
        return general_train_params, bg_param, hp_param, kps_param, train_bg, train_hp, train_kps
