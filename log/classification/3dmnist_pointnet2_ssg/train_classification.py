"""
Author: Benny
Date: Nov 2019
"""

import argparse
import csv
import datetime
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
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
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument(
        '--dataset',
        type=str,
        default='modelnet40',
        choices=['modelnet40', 'modelnet10', '3dmnist'],
        help='dataset name',
    )
    parser.add_argument('--data_path', type=str, default=None, help='dataset root path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1 if platform.system() == 'Darwin' else 2,
        help='number of dataloader workers',
    )
    return parser.parse_args()


def inplace_relu(module):
    classname = module.__class__.__name__
    if classname.find('ReLU') != -1:
        module.inplace = True


def augment_point_cloud(points):
    batch_size, num_points, _ = points.shape

    for batch_index in range(batch_size):
        dropout_ratio = torch.rand(1).item() * 0.875
        drop_mask = torch.rand(num_points) <= dropout_ratio
        if drop_mask.any():
            first_point = points[batch_index, 0, :].clone()
            points[batch_index, drop_mask, :] = first_point

    scales = torch.empty(batch_size, device=points.device, dtype=points.dtype).uniform_(0.8, 1.25)
    shifts = torch.empty(batch_size, 3, device=points.device, dtype=points.dtype).uniform_(-0.1, 0.1)
    points[:, :, 0:3] = points[:, :, 0:3] * scales.view(-1, 1, 1)
    points[:, :, 0:3] = points[:, :, 0:3] + shifts.view(-1, 1, 3)
    return points


def evaluate(model, loader, num_class=40, device=torch.device('cpu')):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if device.type != 'cpu':
            points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1).contiguous()
        pred, _ = classifier(points)
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
        return {
            'name': 'modelnet40',
            'root': args.data_path or 'data/modelnet40_normal_resampled/',
            'num_class': 40,
            'loader': ModelNetDataLoader,
            'copy_files': ['data_utils/ModelNetDataLoader.py'],
        }
    if dataset_name == 'modelnet10':
        return {
            'name': 'modelnet10',
            'root': args.data_path or 'data/modelnet40_normal_resampled/',
            'num_class': 10,
            'loader': ModelNetDataLoader,
            'copy_files': ['data_utils/ModelNetDataLoader.py'],
        }
    if dataset_name == '3dmnist':
        return {
            'name': '3dmnist',
            'root': args.data_path or '3D MNIST',
            'num_class': 10,
            'loader': ThreeDMNISTDataLoader,
            'copy_files': ['data_utils/ThreeDMNISTDataLoader.py', 'data_utils/ModelNetDataLoader.py'],
        }
    raise ValueError(f'Unsupported dataset: {args.dataset}')


def create_dataset(dataset_spec, args, split):
    if dataset_spec['loader'] is ModelNetDataLoader:
        return dataset_spec['loader'](
            root=dataset_spec['root'],
            args=args,
            split=split,
            process_data=args.process_data,
        )
    return dataset_spec['loader'](root=dataset_spec['root'], args=args, split=split)


def get_effective_num_workers(args, dataset_name):
    if dataset_name == '3dmnist':
        return 0
    return args.num_workers


def get_git_metadata():
    try:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        return {'branch': branch, 'commit': commit}
    except Exception:
        return {'branch': 'unknown', 'commit': 'unknown'}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def append_metrics_row(path, row):
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def copy_experiment_sources(exp_dir, args, dataset_spec):
    shutil.copy(f'./models/{args.model}.py', str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))
    shutil.copy('./test_classification.py', str(exp_dir))
    shutil.copy('./pyproject.toml', str(exp_dir))
    for relative_path in dataset_spec['copy_files']:
        shutil.copy(relative_path, str(exp_dir))


def main(args):
    dataset_spec = get_dataset_spec(args)
    args.num_category = dataset_spec['num_class']
    effective_num_workers = get_effective_num_workers(args, dataset_spec['name'])

    def log_string(message):
        logger.info(message)
        print(message)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if effective_num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy('file_system')
        except (AttributeError, RuntimeError):
            pass

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir or timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir.joinpath(f'{args.model}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    device = get_device(use_cpu=args.use_cpu)
    log_string('Using device: %s' % get_device_name(device))
    log_string(f'Dataset: {dataset_spec["name"]}')
    log_string(f'Dataset root: {dataset_spec["root"]}')
    if effective_num_workers != args.num_workers:
        log_string(f'Overriding num_workers from {args.num_workers} to {effective_num_workers} for HDF5 safety')

    log_string('Load dataset ...')
    train_dataset = create_dataset(dataset_spec, args, 'train')
    test_dataset = create_dataset(dataset_spec, args, 'test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
    )

    num_class = dataset_spec['num_class']
    model = importlib.import_module(args.model)
    copy_experiment_sources(exp_dir, args, dataset_spec)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if device.type != 'cpu':
        classifier = classifier.to(device)
        criterion = criterion.to(device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    metrics_path = exp_dir.joinpath('metrics.csv')
    run_config_path = exp_dir.joinpath('run_config.json')
    summary_path = exp_dir.joinpath('summary.json')

    run_metadata = {
        'timestamp': timestr,
        'model': args.model,
        'dataset': dataset_spec['name'],
        'dataset_root': dataset_spec['root'],
        'num_class': num_class,
        'num_point': args.num_point,
        'batch_size': args.batch_size,
        'epochs': args.epoch,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'weight_decay': args.decay_rate,
        'use_normals': args.use_normals,
        'use_uniform_sample': args.use_uniform_sample,
        'process_data': args.process_data,
        'requested_num_workers': args.num_workers,
        'effective_num_workers': effective_num_workers,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'device': get_device_name(device),
        'args': vars(args),
        'git': get_git_metadata(),
    }
    write_json(run_config_path, run_metadata)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0

    try:
        checkpoint = torch.load(checkpoints_dir.joinpath('best_model.pth'), map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_instance_acc = checkpoint.get('instance_acc', 0.0)
        best_class_acc = checkpoint.get('class_acc', 0.0)
        best_epoch = checkpoint.get('epoch', 0)
        log_string(f'Use pretrain model from epoch {start_epoch}')
        log_string(f'Best Instance Accuracy: {best_instance_acc:.4f}, Class Accuracy: {best_class_acc:.4f}')
    except FileNotFoundError:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    except Exception as exc:
        log_string(f'Error loading checkpoint: {type(exc).__name__}: {exc}')
        log_string('Starting training from scratch...')
        start_epoch = 0

    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for _, (points, target) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad(set_to_none=True)

            points = augment_point_cloud(points.float())
            points = points.transpose(2, 1).contiguous()

            if device.type != 'cpu':
                points, target = points.to(device), target.to(device)

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct) if mean_correct else 0.0
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = evaluate(classifier.eval(), test_loader, num_class=num_class, device=device)

        is_best = instance_acc >= best_instance_acc
        if is_best:
            best_instance_acc = instance_acc
            best_epoch = epoch + 1
        if class_acc >= best_class_acc:
            best_class_acc = class_acc

        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

        append_metrics_row(
            metrics_path,
            {
                'epoch': epoch + 1,
                'train_instance_acc': train_instance_acc,
                'test_instance_acc': instance_acc,
                'test_class_acc': class_acc,
                'best_instance_acc': best_instance_acc,
                'best_class_acc': best_class_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'global_step': global_step,
            },
        )

        latest_state = {
            'epoch': epoch + 1,
            'instance_acc': instance_acc,
            'class_acc': class_acc,
            'best_instance_acc': best_instance_acc,
            'best_class_acc': best_class_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(latest_state, checkpoints_dir.joinpath('latest_model.pth'))

        if is_best:
            logger.info('Save model...')
            savepath = checkpoints_dir.joinpath('best_model.pth')
            log_string('Saving at %s' % savepath)
            best_state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'best_instance_acc': best_instance_acc,
                'best_class_acc': best_class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(best_state, savepath)

        global_epoch += 1

        if device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.empty_cache()

    write_json(
        summary_path,
        {
            **run_metadata,
            'best_epoch': best_epoch,
            'best_instance_acc': best_instance_acc,
            'best_class_acc': best_class_acc,
        },
    )
    logger.info('End of training...')


if __name__ == '__main__':
    main(parse_args())
