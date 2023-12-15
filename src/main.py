""" Main script to combine functionalities like dataloading, model building, training, and evaluation. """

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import logging
import argparse

from src.data import Dataset_Setup, AlzDataset
from src.models.resnet50 import Resnet50
from src.train import train_fn
from src.utils import disp_image

def main(
        data_dir,
        url,
        torch_model,
        num_epochs=10,
        batch_size=50,
        learning_rate=0.001,
        train_criterion=torch.nn.CrossEntropyLoss,
        model_optimizer=torch.optim.Adam,
        data_augmentations=None,
        save_model_str=None,
        splits = [0.8, 0.1, 0.1],
        use_all_data_to_train=False,
        exp_name=''):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data setup
    logging.info("Setting up data...")

    # Download and extract data
    Dataset_Setup(data_dir, url)

    # Create datasets
    dataset = AlzDataset(
        data_dir, 
        split={'train': True, 'val': True, 'test': True},
        split_ratio={'train': 0.8, 'val': 0.1, 'test': 0.1},
        transform=data_augmentations)

    train_dataset, val_dataset, test_dataset = dataset.split_dataset()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")

    disp_image(train_dataset[0][0], train_dataset[0][1])


if __name__ == '__main__':

    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}

    opti_dict = {
        'sgd': torch.optim.SGD, 
        'adam': torch.optim.Adam, 
        'rmsprop': torch.optim.RMSprop}
    
    models_dict = {'resnet50': Resnet50}

    parser = argparse.ArgumentParser(description='Alzheimer\'s Disease Classification')
    parser.add_argument('--data_dir', '-D',
                        type=str, 
                        default='data', 
                        help='Directory where data is stored')
    parser.add_argument('--url', '-u',
                        type=str, 
                        default='', 
                        help='URL to download data')
    parser.add_argument('--model', '-m',
                        type=str, 
                        default='Resnet50', 
                        choices=list(models_dict.keys()),
                        help='Torch model to use')
    parser.add_argument('--num_epochs', '-e',
                        type=int, 
                        default=10, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', '-b',
                        type=int, 
                        default=50, 
                        help='Batch size')
    parser.add_argument('--learning_rate', '-l',
                        type=float, 
                        default=0.001, 
                        help='Learning rate')
    parser.add_argument('--training_loss', '-L',
                        type=str, 
                        default='cross_entropy',
                        choices=list(loss_dict.keys()),
                        help='Training criterion')
    parser.add_argument('--optimizer', '-o',
                        type=str, 
                        default='adam',
                        choices=list(opti_dict.keys()),
                        help='Model optimizer')
    parser.add_argument('--data_augmentations', '-d',
                        type=str, 
                        default=None, 
                        help='Data augmentations')
    parser.add_argument('--model_path', '-p',
                        type=str, 
                        default=None, 
                        help='Save model string')
    parser.add_argument('--splits', '-s',
                        nargs='+', 
                        type=float, 
                        default=[0.8, 0.1, 0.1], 
                        help='Train, Val, Test splits')
    parser.add_argument('--use_all_data_to_train', '-a',
                        type=bool, 
                        default=False, 
                        help='Use all data to train')
    parser.add_argument('--exp_name', '-n',
                        type=str, 
                        default='', 
                        help='Experiment name')
    
    args, unknowns = parser.parse_known_args()

    log_lvl = logging.INFO
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        url=args.url,
        torch_model=models_dict[args.model],
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        data_augmentations=eval(args.data_augmentations),
        save_model_str=args.model_path,
        splits=args.splits,
        use_all_data_to_train=args.use_all_data_to_train,
        exp_name=args.exp_name
    )

