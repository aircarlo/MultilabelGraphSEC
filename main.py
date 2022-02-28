import os
import argparse
import yaml
from datetime import datetime
from data.graph_data import create_graph
from net_models.crnn_ggnn_net import CRNN_GGNN
from net_models.cnn_net import VGGLike
from net_models.crnn_net import CRNN
from helpers import get_train_data, get_test_data, start_train, start_test

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="params.cfg", help="path to file containing the parameters")
parser.add_argument("--mode", type=str, help="train or test")

if __name__ == '__main__':
    print(f'Start: - {datetime.now().strftime("%D  %H:%M:%S")}\n')
    # parse main parameters
    cli_args = parser.parse_args()
    # parse other parameters from file
    with open(cli_args.cfg_file, "r") as fd:
        file_args = yaml.load(fd, yaml.FullLoader)
    file_args['mode'] = cli_args.mode

    if file_args['verbose']:
        print('Parameters:')
        for k, v in file_args.items():
            print(k, '->', v)
        print('')
    if file_args['mixup'] and file_args['spec_augm']:
        raise ValueError('mixup and spec_augm cannot be used at the same time!')

    if file_args['arch'] == 'CNN':
        model = VGGLike()
    elif file_args['arch'] == 'CRNN':
        model = CRNN()
    elif file_args['arch'] == 'CRNN-GGNN':
        model = CRNN_GGNN(emb_dim=128)  # emb_dim = fusion level embedding (must equal nodes embedding dimension)
    else:
        raise Exception('unrecognized network model', file_args['arch'])

    # start Test
    if file_args['mode'] == 'test':
        melspec_test_dataset = get_test_data(file_args)
        if os.path.isfile(file_args['graph_embedding']):
            graph_dataset = create_graph(file_args['graph_embedding'], verbose=file_args['verbose'])
        else:
            graph_dataset=None
        start_test(model,
                   melspec_test_dataset,
                   graph_dataset,
                   file_args)

    # start Train
    elif file_args['mode'] == 'train':
        if not os.path.exists(file_args['logs_path']):
            os.makedirs(file_args['logs_path'])
        with open(os.path.join(file_args['logs_path'], 'dump_params.cfg'), 'w') as file:
            yaml.dump(file_args, file)
        # generate dataloaders
        melspec_train_dataloader = get_train_data(file_args)
        if os.path.isfile(file_args['graph_embedding']):
            graph_dataset = create_graph(file_args['graph_embedding'], verbose=file_args['verbose'])
        else:
            graph_dataset=None
        # start train
        start_train(model,
                    melspec_train_dataloader,
                    graph_dataset,
                    file_args)

    elif file_args['mode'] == 'debug':
        print('debug mode')

    else:
        print('ERROR:  --mode argument required, train or test')

    print(f'End: - {datetime.now().strftime("%D  %H:%M:%S")}')
