import json
import os
import shutil
import time
from shutil import copyfile

import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from contrast import models
from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.option import parse_option
from contrast.util import AverageMeter, cosine_scheduler
from contrast.lars import add_weight_decay, LARS
from contrast.loss import TripletLoss


def build_model(args, init_lr):
    encoder = resnet.__dict__[args.arch]
    model = models.__dict__[args.model](encoder, args).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,)
    elif args.optimizer == 'lars':
        params = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.SGD(
            params,
            lr=init_lr,
            momentum=args.momentum,)
        optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], broadcast_buffers=False)
    return model, optimizer


def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(
        f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(args, model, optimizer, scaler, sampler=None):
    logger.info(f"=> loading checkpoint '{args.resume}'")

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

    logger.info(
        f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scaler, sampler=None):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))


def main(args):
    train_prefix = 'train'
    train_loader = get_loader(
        args.aug, args,
        two_crop=args.model in ['CLoVE'],
        prefix=train_prefix,
        return_coord=True,)

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    # ============ init schedulers  ============
    global_learning_rate = args.batch_size * dist.get_world_size() / 256 * \
        args.base_learning_rate * args.grad_accumulation_steps
    logger.info(f"global learning rate: {global_learning_rate}")

    lr_schedule = cosine_scheduler(
        global_learning_rate,  # linear scaling rule
        global_learning_rate * 0.01,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epoch,
        start_warmup_value=global_learning_rate / args.warmup_multiplier
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.clove_momentum, 1.0,
                                         args.epochs, len(train_loader))

    logger.info(f"initial learning rate: {lr_schedule[0]}")
    model, optimizer = build_model(args, init_lr=lr_schedule[0])

    # tensorboard
    scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    if args.pretrained_model:
        assert os.path.isfile(args.pretrained_model)
        load_pretrained(model, args.pretrained_model)
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(
                f'no checkpoint found in {args.output_dir}, ignoring auto resume')
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer, scaler,
                        sampler=train_loader.sampler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    triplet_loss = TripletLoss(args.lamb, args.clove_pos_ratio).cuda()
    torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, lr_schedule,
              momentum_schedule, triplet_loss, args, scaler, summary_writer)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model, optimizer,
                            scaler, sampler=train_loader.sampler)


def add_images_to_tensorboard(writer, iter, images, tag_name):
    grid = torchvision.utils.make_grid(images.unsqueeze(1))
    writer.add_image(tag_name, grid, iter, dataformats='CHW')


def train(epoch, train_loader, model, optimizer, lr_schedule, momentum_schedule, triplet_loss, args, scaler, summary_writer):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):

        images = data[0]
        coords = data[1]

        images = [item.cuda(non_blocking=True) for item in images]
        coords = [item.cuda(non_blocking=True) for item in coords]

        step = (epoch - 1) * len(train_loader) + idx

        # get learning rate and momentum
        lr = lr_schedule[step]
        m = momentum_schedule[step]

        sync_gradients = ((idx + 1) % args.grad_accumulation_steps ==
                          0) or (idx + 1 == len(train_loader))

        if not sync_gradients:
            with model.no_sync():
                with torch.cuda.amp.autocast(True):
                    gl_sa_maps, lo_sa_maps, gl_projs_ng, gl_sa_weights, lo_sa_weights = model(
                        images)

                loss = triplet_loss(
                    gl_sa_maps.float(), lo_sa_maps.float(), gl_projs_ng.float(), coords)
                loss /= args.grad_accumulation_steps
                scaler.scale(loss).backward()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Gradients finally sync
            with torch.cuda.amp.autocast(True):
                gl_sa_maps, lo_sa_maps, gl_projs_ng, gl_sa_weights, lo_sa_weights = model(
                    images, m)

            loss = triplet_loss(
                gl_sa_maps.float(), lo_sa_maps.float(), gl_projs_ng.float(), coords)

            loss /= args.grad_accumulation_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update meters and print info
        loss_meter.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')

            # tensorboard logger
            if summary_writer is not None:
                summary_writer.add_scalar('metrics/m', m, step)
                summary_writer.add_scalar('metrics/lr', lr, step)
                summary_writer.add_scalar('loss/total', loss.item(), step)
                add_images_to_tensorboard(
                    summary_writer, step, gl_sa_weights, "masks/gl_sa_weights")
                add_images_to_tensorboard(
                    summary_writer, step, lo_sa_weights, "masks/lo_sa_weights")


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    opt.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir,
                          distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        shutil.copyfile(
            "./main_pretrain.py", os.path.join(opt.output_dir,
                                               "main_pretrain.py")
        )
        shutil.copyfile(
            "./contrast/models/CLoVE.py", os.path.join(
                opt.output_dir, "CLoVE.py")
        )
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(opt)).items()))
    )

    main(opt)
