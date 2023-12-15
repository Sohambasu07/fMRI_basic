""" Main script to combine functionalities like dataloading, model building, training, and evaluation. """

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import lr_scheduler
from torchsummary import summary
import logging
import argparse
import os

from src.data import Dataset_Setup, AlzDataset
from src.models.resnet50 import Resnet50
from src.train import train_fn
from src.eval import eval_fn
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
        load_model=False,
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
    
    print(f"Total Dataset size: {len(dataset)}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split_dataset()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    channels, img_height, img_width = train_dataset[0][0].shape

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Image shape: ({channels} x {img_height} x {img_width})")

    # Model setup
    model = torch_model()
    if load_model:
        model.load_state_dict(torch.load(save_model_str))
    model.to(device)

    # Print model summary
    summary(model, (channels, img_height, img_width), device=device)

    # instantiate optimizer
    optimizer = model_optimizer(model.parameters(), lr=learning_rate)

    # LR scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.33)

    # Train model

    logging.info("Training model...")
    best_val_acc = 0

    for epoch in range(num_epochs):

        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_acc, train_loss = train_fn(
            model, 
            optimizer, 
            train_criterion, 
            train_loader, 
            device)
        logging.info('Training accuracy: %f', train_acc)
        logging.info('Training loss: %f', train_loss)
        scheduler.step()

        val_acc, val_loss = eval_fn(
            model, 
            train_criterion, 
            val_loader, 
            device)
        logging.info('Validation accuracy: %f', val_acc)
        logging.info('Validation loss: %f', val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_model_str:
            # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
                model_save_dir = os.path.join(os.getcwd(), save_model_str)

                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                # save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + str(int(time.time())))
                save_path = os.path.join(model_save_dir, 'best_model.pt')
                torch.save(model.state_dict(), save_path)

                print('Validation accuracy increased to %f, saving model to %s' %(val_acc, save_path))



if __name__ == '__main__':

    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}

    opti_dict = {
        'sgd': torch.optim.SGD, 
        'adam': torch.optim.Adam, 
        'rmsprop': torch.optim.RMSprop}
    
    models_dict = {'Resnet50': Resnet50}

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
    # parser.add_argument('--data_augmentations', '-d',
    #                     type=str, 
    #                     default=None, 
    #                     help='Data augmentations')
    parser.add_argument('--model_path', '-p',
                        type=str, 
                        default='./saved_models/best_model.pt', 
                        help='Save model string')
    parser.add_argument('--load_model', '-ld',
                        type=bool, 
                        default=False, 
                        help='Load a saved model')
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
        # data_augmentations=eval(args.data_augmentations),
        save_model_str=args.model_path,
        load_model=args.load_model,
        splits=args.splits,
        use_all_data_to_train=args.use_all_data_to_train,
        exp_name=args.exp_name
    )

