from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import time

from torchvision.transforms.transforms import ColorJitter
from opts import parse_opts
from model_snn_cnn import generate_model_snn, make_data_parallel
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from datasets import get_training_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
import inference
from torchvision.models import resnet18


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.event_video_path = opt.root_path / opt.event_video_path
        opt.frame_video_path = opt.root_path / opt.frame_video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1

    opt.event_mean, opt.event_std = get_mean_std(opt.value_scale, data_type='event')
    opt.frame_mean, opt.frame_std = get_mean_std(opt.value_scale, data_type='frame')

    opt.n_input_channels = 3

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):

    
    
    event_spatial_transform = [Resize(opt.sample_size)]
    frame_spatial_transform = [Resize(opt.sample_size)]
    event_spatial_transform.append(ToTensor())
    frame_spatial_transform.append(ToTensor())
    event_spatial_transform.append(ScaleValue(opt.value_scale))
    frame_spatial_transform.append(ScaleValue(opt.value_scale))
    
    
    event_spatial_transform = Compose(event_spatial_transform)
    frame_spatial_transform = Compose(frame_spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.event_video_path,
                                   opt.frame_video_path, opt.annotation_path,
                                   event_spatial_transform, frame_spatial_transform, temporal_transform)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)

def get_inference_utils(opt):
    
    
    event_spatial_transform = [Resize(opt.sample_size)]
    frame_spatial_transform = [Resize(opt.sample_size)]
    event_spatial_transform.append(ToTensor())
    frame_spatial_transform.append(ToTensor())
    event_spatial_transform.append(ScaleValue(opt.value_scale))
    
    frame_spatial_transform.append(ScaleValue(opt.value_scale))
    
    event_spatial_transform = Compose(event_spatial_transform)
    frame_spatial_transform = Compose(frame_spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(TemporalRandomCrop(opt.inference_sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    inference_data, collate_fn = get_inference_data(
        opt.event_video_path, opt.frame_video_path, opt.annotation_path, opt.inference_subset, event_spatial_transform, frame_spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        print('look look look')
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    os.environ['PYTHONHASHSEED'] = str(opt.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    opt.is_master_node = not opt.distributed or opt.dist_rank == 0
    model = generate_model_snn()

    model = make_data_parallel(model, opt.distributed, opt.device)
    if opt.is_master_node:
        print(model)
    parameters = model.parameters()
    criterion = CrossEntropyLoss().to(opt.device)

    (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)

    # 🔴🔴🔴 Retinex 可视化验证（仅首次运行时取消下方注释；生成图片后立即重新注释！）
    """if opt.is_master_node:
        print("\n🔍 正在生成 Retinex 验证图...")
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                frame_inputs = batch[0].to(opt.device)  # [B, 3, T, H, W]
                sample_frame = frame_inputs[0, :, 0].unsqueeze(0)  # [1, 3, 180, 180]
                break
            if hasattr(model, 'module'):
                reflectance = model.module.retinex_processor(sample_frame)
            else:
                reflectance = model.retinex_processor(sample_frame)
            import torchvision.utils as vutils
            vutils.save_image(sample_frame, opt.result_path / "retinex_original.png", normalize=False)
            vutils.save_image(reflectance, opt.result_path / "retinex_output.png", normalize=False)
            print(f"✅ 已保存: {opt.result_path}/retinex_original.png & retinex_output.png")
        model.train()
    #    raise SystemExit("✅ Retinex 验证完成！请检查图片后删除或注释以上代码块！")
    # 🔴🔴🔴 END Retinex verification"""

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)
        scheduler.step()
        if opt.inference and i >=40 and (i - 40) % 3 == 0:
            print(f"▶ Running inference at epoch {i}...")
            for test_num in range(10):
                inference_loader, inference_class_names = get_inference_utils(opt)
                inference_result_path = opt.result_path / f'test_epoch{i}_run{test_num+1}.json'
                inference.inference(
                    inference_loader, model, inference_result_path,
                    inference_class_names,
                    opt.inference_no_average,
                    opt.output_topk,
                    i, tb_writer, opt.distributed, opt.device, str(test_num + 1)
                )


if __name__ == '__main__':
    opt = get_opt()
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    print('opt.ngpus_per_node',opt.ngpus_per_node)
    main_worker(-1, opt)