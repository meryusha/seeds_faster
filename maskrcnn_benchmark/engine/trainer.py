# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data import make_data_loader
import numpy as np
import tensorflow as tf
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments, cfg,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    summary_writer = tf.summary.FileWriter(cfg.OUTPUT_DIR)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    dataset_names = cfg.DATASETS.TEST
    err_arr = [1]
    mAP_arr = [0]
    thresh_mAP = 0
    thresh_err = 0.05
    platueIters = 0
    print('start iter', start_iter)
    print('len of DL', len(data_loader))
    valid_folder = os.path.join(cfg.OUTPUT_DIR, "valid")
    mkdir(valid_folder)
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        summaries = [
            tf.Summary.Value(
                tag="Train/Loss",
                simple_value=losses.item()),
        ]
        summaries.extend([
            tf.Summary.Value(tag=f"Learning_rate/lr_{i}", simple_value=lr)
            for i, lr in enumerate(scheduler.get_lr())
        ])


        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                mAP, error_seed, error_rad = inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types = ("bbox",),
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                )
                summaries.extend([
                    tf.Summary.Value(tag="Valid/mAP", simple_value = mAP),
                ])
                summaries.extend([
                    tf.Summary.Value(tag='error_seed', simple_value =  error_seed),
                ])
                summaries.extend([
                    tf.Summary.Value(tag='error_radical', simple_value =  error_rad),
                ])
                err_mean = (error_seed+ error_rad) / 2.0
                if (mAP - mAP_arr[-1]) < thresh_mAP and (err_mean - err_arr[-1]) > thresh_err and mAP >= 0.8 and err_mean < 0.07:
                    logger.info("PLATUE")
                    platueIters += 1 
                else:
                    platueIters = 0
                   
                mAP_arr.append(mAP)
                err_arr.append(err_mean)

                np.save(os.path.join(valid_folder, f"error_seed_{iteration}.npy"), error_seed)
                np.save(os.path.join(valid_folder, f"error_radical_{iteration}.npy"), error_rad)


                if platueIters >= 7:
                    logger.info("Early stopping")
                    checkpointer.save("model_final", **arguments)
                    summary_writer.add_summary(tf.Summary(value=summaries), iteration)
                    break

                print('mAP', mAP)
                print('error_seed', error_seed)
                print('error_rad', error_rad)

            model.train()
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        summary_writer.add_summary(tf.Summary(value=summaries), iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    summary_writer.flush()
    summary_writer.close()
