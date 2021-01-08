import argparse


# import json
# import torch
# import tarfile
# import pickle
# import matplotlib.pyplot as plt
# import torchvision as tv
# import pathlib                          # Path management tool (standard library)
# import subprocess                       # Runs shell commands via Python (standard library)
# import sagemaker                        # SageMaker Python SDK
# from sagemaker.pytorch import PyTorch   # PyTorch Estimator for TensorFlow


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('-w', '--workers', default=4, type=int, help='Num workers')
    args = parser.parse_args()

    args.data_dir = "/opt/ml/input/data"
    args.workers = 64
    data_dir = args.data_dir
    num_workers = args.workers

    print(data_dir, num_workers, vars(args))


if __name__ == '__main__':
    main()
