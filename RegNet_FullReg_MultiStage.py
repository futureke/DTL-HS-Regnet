import argparse
import datetime
import json
import logging
from pathlib import Path
import shutil
import functions.elastix_python as elx
import functions.general_utils as gut
import functions.landmarks_utils as lu
import functions.network as network
import functions.registration as reg
import functions.setting.setting_utils as su
from functions.setting.experiment_setting import load_network_multi_stage_from_predefined


def run():

    exp_val_list = [
        {'exp': '2020_multistage_crop4_K_NoResp_more_itr', 'slist': [4, 2, 1], 'BaseReg': 'Affine'},
        {'exp': '2020_multistage_crop4_K_NoResp_more_itr', 'slist': [4, 2], 'BaseReg': 'Affine'},
        {'exp': '2020_multistage_crop4_K_NoResp_more_itr', 'slist': [4], 'BaseReg': 'Affine'},
        ]

    current_experiment_loop = exp_val_list
    for current_exp_dict in current_experiment_loop:
        full_registration_multi_stage(current_exp_dict)


def full_registration_multi_stage(exp_dict):


    current_experiment = exp_dict.get('exp', 'elx_registration')
    stage_list = exp_dict['slist']
    setting, backup_folder = initialize(current_experiment, stage_list)
    landmark_calculation = False
    overwrite_dvf = False
    overwrite_landmarks = False
    overwrite_landmarks_hard = False
    setting['WriteNoDVF'] = False
    setting['WriteMasksForLSTM'] = False

    setting['BaseReg'] = exp_dict.get('BaseReg', 'Affine')
    setting['read_pair_mode'] = exp_dict.get('read_pair_mode', 'real')

    setting['reverse_order'] = False
    read_pair_mode_stage = 1
    if setting['read_pair_mode'] == 'synthetic':
        setting['WriteMasksForLSTM'] = True
        setting['WriteNoDVF'] = True
        train_mode = 'Training+Validation'
        setting['NetworkValidation'] = {'NumberOfImagesPerChunk':  10}
    else:
        train_mode = 'Testing'  # 'Training', ' Validation', 'Testing'
    setting['use_keras'] = False


    data_dir_4d = [
        {'data': 'DIR-Lab_4D',
         'TestingCNList': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         'TestingPairList': [[0, 5]],
         'TestingSpacing': 'RS1',  # 'RS1', 'Raw'
         'TrainingCNList': [1, 2, 3],
         'TrainingPairList': [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
         }
    ]

    if setting['read_pair_mode'] == 'synthetic':
        dsmoothlist_training, dsmoothlist_validation = su.dsmoothlist_by_deform_exp(exp_dict['deform_exp'], exp_dict['AGMode'])
        data_exp_dict = [{'data': 'SPREAD',
                          'deform_exp': exp_dict['deform_exp'],
                          'TrainingCNList': [i for i in range(1, 11)],
                          'TrainingTypeImList': [0, 1],
                          'TrainingDSmoothList': su.repeat_dsmooth_numbers(dsmoothlist_training, exp_dict['deform_exp'], repeat=2),
                          'TrainingDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                          'ValidationCNList': [11, 12],
                          'ValidationTypeImList': [0, 1],
                          'ValidationDSmoothList': dsmoothlist_validation,
                          'ValidationDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                          },
                         {'data': 'DIR-Lab_COPD',
                          'deform_exp': exp_dict['deform_exp'],
                          'TrainingCNList': [i for i in range(1, 10)],
                          'TrainingTypeImList': [0, 1],
                          'TrainingDSmoothList': su.repeat_dsmooth_numbers(dsmoothlist_training, exp_dict['deform_exp'], repeat=2),
                          'TrainingDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                          'ValidationCNList': [10],
                          'ValidationTypeImList': [0, 1],
                          'ValidationDSmoothList': dsmoothlist_validation,
                          'ValidationDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                          }
                         ]
    else:
        data_exp_dict = data_dir_4d

    setting = su.load_setting_from_data_dict(setting, data_exp_dict)

    network_dict = load_network_multi_stage_from_predefined(current_experiment)

    setting['normalization'] = ''
    setting['ImagePyramidSchedule'] = stage_list
    setting['PaddingForDownSampling'] = 'constant'
    setting['WriteAfterEachStage'] = True
    setting['verbose'] = True
    setting['verbose_image'] = True
    setting['UseTorsoMask'] = True
    setting['UseMask'] = True

    setting['Reg_Affine_Parameter'] = 'Par0011.affine.txt'
    setting['Reg_BSpline_Parameter'] = 'parameters_MI_BSpline_500.txt'
    setting['Reg_BSpline_Folder'] = bspline_folder_by_stage_list(stage_list)
    setting['Reg_Affine_Mask'] = 'Torso'
    setting['Reg_BSpline_Mask'] = 'Lung'
    setting['Reg_NumberOfThreads'] = 7

    network_dict = get_parameter_multi_stage_network(setting, network_dict)
    setting['PadTo'] = padto_multi_stage(network_dict)
    setting['network_dict'] = network_dict

    if stage_list not in [[4, 2, 1], [4, 2], [4], [2, 1], [2], [1]]:
        raise ValueError('In the current implementation stage_list can be only be selected between: '
                         '[4, 2, 1]: three stages, [2, 1]: two stages and [1]: one stages ')

    su.write_setting(setting, setting_address=backup_folder + 'setting.txt')
    with open(backup_folder + 'network.txt', 'w') as file:
        file.write(json.dumps(setting['network_dict'], sort_keys=True, indent=4, separators=(',', ': ')))

    pair_info_list = su.get_pair_info_list_from_train_mode_random(setting, train_mode=train_mode, stage=read_pair_mode_stage,
                                                                  load_mode='Pair')
    for i_pair_info, pair_info in enumerate(pair_info_list):
        reg.multi_stage(setting, pair_info=pair_info, overwrite=overwrite_dvf)
        if landmark_calculation:
            lu.calculate_write_landmark(setting, pair_info=pair_info, overwrite_landmarks=overwrite_landmarks,
                                        overwrite_landmarks_hard=overwrite_landmarks_hard, overwrite_bspline_dvf=True, )
        logging.debug('RegNet3D_FullReg_MultiStage, pair {}/{} is done'.format(i_pair_info, len(pair_info_list)-1))


def get_parameter_multi_stage_network(setting, network_dict):
    if network_dict is not None:
        for key in network_dict.keys():
            net = network_dict[key]
            if net['Ry_erode'] == 'Auto':
                if net['NetworkDesign'] in ['crop5_connection', 'unet1']:
                    net['Ry_erode'] = 0
                else:
                    net['Ry_erode'] = 2
            net['GlobalStepLoad'] = su.get_global_step(setting, net['GlobalStepLoad'], net['NetworkLoad'])

    return network_dict


def padto_multi_stage(network_dict):
    padto_setting = dict()
    if network_dict is not None:
        for stage_key in network_dict.keys():
            padto_setting[stage_key] = getattr(getattr(network, network_dict[stage_key]['NetworkDesign']), 'get_padto')()
    return padto_setting


def bspline_folder_by_stage_list(stage_list):
    bspline_folder = ''
    for stage_str in stage_list:
        bspline_folder = bspline_folder + 'S' + str(stage_str) + '_'
    bspline_folder = bspline_folder[:-1]
    return bspline_folder


def initialize(current_experiment, stage_list, folder_script='functions'):
    parser = argparse.ArgumentParser(description='read where_to_run')
    parser.add_argument('--where_to_run', '-w',
                        help='This is an optional argument, '
                             'you choose between "Auto" or "Cluster". The default value is "Auto"')
    args = parser.parse_args()
    where_to_run = args.where_to_run

    setting = su.initialize_setting(current_experiment=current_experiment, where_to_run=where_to_run)
    date_now = datetime.datetime.now()
    backup_number = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'. \
        format(date_now.year, date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)
    backup_root_folder = su.address_generator(setting, 'result_step_folder', stage_list=stage_list) + 'CodeBackup/'
    backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    gut.logger.set_log_file(backup_folder + 'log.txt', short_mode=True)
    shutil.copy(Path(__file__), Path(backup_folder) / Path(__file__).name)
    shutil.copytree(Path(__file__).parent / Path(folder_script), Path(backup_folder) / Path(folder_script))
    return setting, backup_folder


if __name__ == '__main__':
    run()
