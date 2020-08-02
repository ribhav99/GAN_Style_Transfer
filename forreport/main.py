import argparse
from attrdict import AttrDict
import train
import trainVAE
import evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args):
    args = AttrDict()
    args_dict = {
        'dis_learning_rate': 0.0002,
        'gen_learning_rate': 0.0002,
        'image_dimensions': (128, 128, 1),
        'cartoon_dimensions': (128, 128, 1),
        'batch_size': 64,  # make sure num_train_samples % batch_size >= 10
        'max_pool': (2, 2),
        'num_epochs': 35,
        'kernel_size': 4,
        'num_train_samples': 70000,
        'padding': 1,  # (kernel_size - 1) //2
        # first entry must match last entry of cartoon dim
        'gen_channels': [1, 32, 64, 128, 256],  # starts 3 for coloured
        'dis_channels': [1, 32, 64, 128, 256],  # starts 3 for coloured
        'num_residual_layers': 6,
        'image_save_f': 1,
        'discrim_train_f': 1,
        'lambda_cycle': 10,
        'dropout': True,
        'decay': True,
        'load_models': False,
        'model_path': "/content/modelFinal140.pt",
        'pool': nn.AvgPool2d,
        'activation': nn.LeakyReLU,
        'norm': nn.InstanceNorm2d,
        'human_train_path': "/content/GAN_Style_Transfer/data/human_train.txt",
        'human_test_path': "/content/GAN_Style_Transfer/data/human_test.txt",
        'cartoon_train_path': "/content/GAN_Style_Transfer/data/cartoon_train.txt",
        'cartoon_test_path': "/content/GAN_Style_Transfer/data/cartoon_test.txt",
        'human_data_root_path': "/content/humangray128/",
        'cartoon_data_root_path': "/content/cartoonfacesgray/",
        'save_path': "/content/GAN_Style_Transfer/Models",
        'use_wandb': True,
        'test': False,
        'buffer_train': True
    }
    args.update(args_dict)

    if not args.test:
        print(
            "---Trains 10 images for 10 epochs to indicate that our training loop works---")
    else:
        print("---Will evaluate the specified trained model now---")
    folder_path = ""
    isVAE = False
    if args.Conv2T:
        args_dict['Conv2T'] = True
        args_dict['norm_type'] = 'batch'
        folder_path = "../conv2T"
    elif args.RegConv:
        args_dict['norm_type'] = 'batch'
        folder_path = "../RegConv"
    elif args.VAE:
        isVAE = True
        folder_path = "../VAE"
    elif (args.Gray):
        pass

    if not args.test:
        train.train(args_dict, device)
    else:
        evaluate.evaluate(folder_path, args_dict, isVAE=isVAE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-Conv2T', action='store_true')
    parser.add_argument('-RegConv', action='store_true')
    parser.add_argument('-VAE', action='store_true')
    parser.add_argument('-Gray', action='store_true')
    parser.add_argument('-train', action='store_true')
    args = parser.parse_args()
    run(args)
