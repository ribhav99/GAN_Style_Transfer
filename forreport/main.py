import argparse
from attrdict import AttrDict
import train2
import trainVAE
import evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args):
    run_args = AttrDict()
    args_dict = {
        'dis_learning_rate': 0.0002,
        'gen_learning_rate': 0.0002,
        'batch_size': 1,
        'num_epochs': 10,
        'human_root_dir': "../trainHuman/",
        'cartoon_root_dir': "../trainCartoon/",
        'act_fn_gen': 'relu',
        'act_fn_dis': 'lrelu',
        'norm_type': 'instance',
        'num_res': 3,
        'dropout': False,
        'lambda_cycle': 10,
        'gray': False,
        'Conv2T': False
    }
    if (args.train):
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
    run_args.update(args_dict)
    if args.train:
        train2.train(run_args, device)
    else:
        evaluate.evaluate(folder_path, run_args, isVAE=isVAE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-Conv2T', action='store_true')
    parser.add_argument('-RegConv', action='store_true')
    parser.add_argument('-VAE', action='store_true')
    parser.add_argument('-Gray', action='store_true')
    parser.add_argument('-train', action='store_true')
    args = parser.parse_args()
    run(args)
