import argparse
from train import train_model
from ibtd_predict import run_prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='predict')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        run_prediction()

if __name__ == '__main__':
    main()
