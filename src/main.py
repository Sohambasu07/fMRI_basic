""" Main script to combine functionalities like dataloading, model building, training, and evaluation. """

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary
import logging
import argparse
import os
from pathlib import Path
import wandb

from src.data import Dataset_Setup, AlzDataset
from src.models.resnet50 import Resnet50
from src.models.mobilenetv3 import MobileNetv3
from src.models.efficientnetb4 import EfficientNetB4
from src.train import train_fn
from src.evaluate import eval_fn

def main(
        data_dir,
        url,
        torch_model,
        num_epochs=10,
        batch_size=64,
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
    model_save_dir = Path(os.getcwd()) / save_model_str
    save_path = model_save_dir / 'best_model.pt'
    model = torch_model()
    if load_model:
        model.load_state_dict(torch.load(save_path))
    model.to(device)

    # Print model summary
    summary(model, (channels, img_height, img_width), device=device.type)

    # instantiate optimizer
    optimizer = model_optimizer(model.parameters(), lr=learning_rate)

    # LR scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.33)

    # Wandb setup
    wandb.login()

    with wandb.init(
        # set the wandb project where this run will be logged
        project="3dqd-pvqvae",
            
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "PVQVAE",
        "dataset": "ShapeNetv2",
        "optimizer": optimizer.__class__.__name__,
        "train_criterion": train_criterion.__class__.__name__,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "lr_schedule": scheduler.__class__.__name__,
        }
    ) as run:

        # Training model
        logging.info("Training model...")
        best_val_loss = 0

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
            wandb.log({"train_acc": train_acc, "train_loss": train_loss})
            scheduler.step()

            #Validation
            val_acc, val_loss = eval_fn(
                model, 
                train_criterion, 
                val_loader, 
                device)
            logging.info('Validation accuracy: %f', val_acc)
            logging.info('Validation loss: %f', val_loss)
            wandb.log({"val_acc": val_acc, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                torch.save(model.state_dict(), save_path)
                best_model = wandb.Artifact(f"best_model_{run.id}", type="model")
                best_model.add_file('./best_model.pt')
                run.log_artifact(best_model)
                logging.info("Best Model saved")
                logging.info(val_loss)
                logging.info(best_val_loss)

                print('Validation Loss decreased to %f, saving model to %s' %(best_val_loss, save_path))
        run.finish()



if __name__ == '__main__':

    loss_dict = {'cross_entropy': torch.nn.BCELoss()}

    opti_dict = {
        'sgd': torch.optim.SGD, 
        'adam': torch.optim.Adam, 
        'rmsprop': torch.optim.RMSprop
        }
    
    models_dict = {
        'Resnet50': Resnet50,
        'MobileNetV3': MobileNetv3,
        'EfficientNetB4': EfficientNetB4
        }

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
                        default=64, 
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
                        default='saved_models', 
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

