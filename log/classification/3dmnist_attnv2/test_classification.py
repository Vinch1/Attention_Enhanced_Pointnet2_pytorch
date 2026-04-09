"""
Author: Benny
Date: Nov 2019
"""

import argparse
import importlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ThreeDMNISTDataLoader import ThreeDMNISTDataLoader
from device_utils import get_device, get_device_name

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument(
        '--dataset',
        type=str,
        default='modelnet40',
        choices=['modelnet40', 'modelnet10', '3dmnist'],
        help='dataset name',
    )
    parser.add_argument('--data_path', type=str, default=None, help='dataset root path')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1 if platform.system() == 'Darwin' else 2,
        help='number of dataloader workers',
    )
    return parser.parse_args()


def evaluate(model, loader, num_class=40, vote_num=1, device=torch.device('cpu')):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if device.type != 'cpu':
            points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1).contiguous()
        vote_pool = torch.zeros(target.size()[0], num_class, device=device if device.type != 'cpu' else None)

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    valid = class_acc[:, 1] > 0
    class_acc[valid, 2] = class_acc[valid, 0] / class_acc[valid, 1]
    class_acc = np.mean(class_acc[valid, 2]) if np.any(valid) else 0.0
    instance_acc = np.mean(mean_correct) if mean_correct else 0.0
    return instance_acc, class_acc


def get_dataset_spec(args):
    dataset_name = args.dataset.lower()
    if dataset_name == 'modelnet40':
        return {'name': 'modelnet40', 'root': args.data_path or 'data/modelnet40_normal_resampled/', 'num_class': 40, 'loader': ModelNetDataLoader}
    if dataset_name == 'modelnet10':
        return {'name': 'modelnet10', 'root': args.data_path or 'data/modelnet40_normal_resampled/', 'num_class': 10, 'loader': ModelNetDataLoader}
    if dataset_name == '3dmnist':
        return {'name': '3dmnist', 'root': args.data_path or '3D MNIST', 'num_class': 10, 'loader': ThreeDMNISTDataLoader}
    raise ValueError(f'Unsupported dataset: {args.dataset}')


def create_dataset(dataset_spec, args):
    if dataset_spec['loader'] is ModelNetDataLoader:
        return dataset_spec['loader'](root=dataset_spec['root'], args=args, split='test', process_data=False)
    return dataset_spec['loader'](root=dataset_spec['root'], args=args, split='test')


def get_effective_num_workers(args, dataset_name):
    if dataset_name == '3dmnist':
        return 0
    return args.num_workers


def main(args):
    experiment_dir = Path('log/classification') / args.log_dir

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(experiment_dir / 'eval.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(message):
        logger.info(message)
        print(message)

    device = get_device(use_cpu=args.use_cpu)
    log_string('Using device: %s' % get_device_name(device))

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_config_path = experiment_dir / 'run_config.json'
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text())
        args.dataset = run_config.get('dataset', args.dataset)
        args.data_path = run_config.get('dataset_root', args.data_path)
        args.num_category = run_config.get('num_class', args.num_category)

    dataset_spec = get_dataset_spec(args)
    args.num_category = dataset_spec['num_class']
    effective_num_workers = get_effective_num_workers(args, dataset_spec['name'])

    log_string('PARAMETER ...')
    log_string(args)
    log_string(f'Dataset: {dataset_spec["name"]}')
    log_string(f'Dataset root: {dataset_spec["root"]}')

    test_dataset = create_dataset(dataset_spec, args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
    )

    model_name = os.listdir(experiment_dir / 'logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
    if device.type != 'cpu':
        classifier = classifier.to(device)

    checkpoint = torch.load(experiment_dir / 'checkpoints' / 'best_model.pth', map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = evaluate(
            classifier.eval(),
            test_loader,
            vote_num=args.num_votes,
            num_class=args.num_category,
            device=device,
        )
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    main(parse_args())
