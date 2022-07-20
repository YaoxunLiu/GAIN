import os
import logging
import torch
from fuxictr import datasets
from fuxictr.datasets.criteo import FeatureEncoder
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DNN, GAIN, DeepFM, AFN, DCN, WideDeep
from fuxictr.pytorch.torch_utils import seed_everything

if __name__ == '__main__':
    # Load params from config files 
    config_dir = 'config'
    experiment_id = 'GAIN_Criteo_x4'
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    # Get train and validation data generator from h5
    train_gen, valid_gen = datasets.h5_generator(feature_map, 
                                                stage='train', 
                                                train_data=os.path.join(data_dir, 'train.h5'),
                                                valid_data=os.path.join(data_dir, 'valid.h5'),
                                                batch_size=params['batch_size'],
                                                shuffle=params['shuffle'])

    # Model initialization and fitting                                           
    model = GAIN(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    model.fit_generator(train_gen, 
                        validation_data=valid_gen, 
                        epochs=params['epochs'],
                        verbose=params['verbose'])

    model.load_weights(model.checkpoint) # reload the best checkpoint

    # Evalution on validation
    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    # Evalution on test
    logging.info('***** test results *****')
    test_gen = datasets.h5_generator(feature_map, 
                                    stage='test',
                                    test_data=os.path.join(data_dir, 'test.h5'),
                                    batch_size=params['batch_size'],
                                    shuffle=False)
    model.evaluate_generator(test_gen)
