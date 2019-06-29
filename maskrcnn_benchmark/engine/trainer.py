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
from maskrcnn_benchmark.data import make_data_loader_AL
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
                err_mean = (error_seed + error_rad) / 2.0
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





def do_train_AL(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments, cfg, indices, is_passive,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    iteration_global = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    summary_writer = tf.summary.FileWriter(cfg.OUTPUT_DIR)    
    err_arr = [1]
    mAP_arr = [0]
    AL_append_new = 5
    thresh_mAP = 0
    thresh_err = 0.05
    platueIters = 0
    # print('start iter', start_iter)
    print('len of DL', len(data_loader))
    valid_folder = os.path.join(cfg.OUTPUT_DIR, "valid")
    mkdir(valid_folder)
    max_iter_global = 17000
    
    isDone = False
    while iteration_global != max_iter_global:
        if isDone:
            data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
        else:
            data_loaders_val = make_data_loader_AL(cfg, is_train=False, is_distributed=False, indices = indices, is_passive = is_passive)
        dataset_names = cfg.DATASETS.TRAIN
        scores_dict = None
        for iteration, (images, targets, _) in enumerate(data_loader):
            data_time = time.time() - end
            iteration_global = iteration_global + 1
            iteration = iteration + 1
            arguments["iteration"] = iteration_global

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
            # summaries.extend([
            #     tf.Summary.Value(tag=f"Learning_rate/lr_{i}", simple_value=lr)
            #     for i, lr in enumerate(scheduler.get_lr())
            # ])

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 50 == 0 or iteration == max_iter:
                for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                    mAP, error_seed, error_rad, scores_dict = inference(
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

                    np.save(os.path.join(valid_folder, f"error_seed_{iteration_global}.npy"), error_seed)
                    np.save(os.path.join(valid_folder, f"error_radical_{iteration_global}.npy"), error_rad)


                    if platueIters >= 7:
                        logger.info("Early stopping")
                        checkpointer.save("model_final", **arguments)
                        summary_writer.add_summary(tf.Summary(value=summaries), iteration_global)
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
                        iter=iteration_global,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration_global % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration_global), **arguments)
            if iteration == max_iter_global:
                checkpointer.save("model_final", **arguments)

            summary_writer.add_summary(tf.Summary(value=summaries), iteration_global)
        if isDone:
            break
        if is_passive:
            len_ind = len(indices)
            if 101 - len_ind < AL_append_new:
                indices = np.append(indices, np.arange(len_ind,  101)) 
                isDone = True 
            else:
                indices = np.append(indices, np.arange(len_ind, len_ind + AL_append_new))
            data_loader = make_data_loader_AL(cfg, is_train=True, is_distributed=False, indices = indices, is_passive = is_passive)
        else:
            if scores_dict is not None:
                if len(scores_dict) < AL_append_new:
                    images_to_add = np.array(list(scores_dict.keys()))
                    isDone = True
                    logger.info("Adding last batch of images!")
                else:
                    avg_worst_scores = np.zeros(shape = (len(scores_dict)))
                    # for i, (k, v) in enumerate(d.items()):
                    image_id_arr = np.array(list(scores_dict.keys()))
                    print('image_id_arr',image_id_arr)
                    
                    for i, (image_id, scores) in enumerate(scores_dict.items()):
                        scores = np.array(scores)
                        if len(scores) >= AL_append_new:
                            idx_smallest = np.argpartition(scores, AL_append_new)
                            smallest_scores = scores[idx_smallest[:AL_append_new]]
                            avg_worst_scores[i] = np.mean(smallest_scores)
                        else:
                            avg_worst_scores[i] = np.mean(scores)
                    print('avg_worst_scores' ,avg_worst_scores)
                    idx_smallest = np.argpartition(avg_worst_scores, AL_append_new)
                    print('idx_smallest', idx_smallest)
                    images_to_add = image_id_arr[idx_smallest[:AL_append_new]]
                # if images_to_add in indices:
                #     logger.info("Error! Adding same images twice!!!")
                indices = np.append(indices, images_to_add)
            else:
                logger.info("Did not get scores, adding random images for AL then")
                len_ind = len(indices)
                indices = np.append(indices, np.arange(len_ind, len_ind + AL_append_new))

            data_loader = make_data_loader_AL(cfg, is_train=True, is_distributed=False, indices = indices, is_passive = is_passive)




    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    summary_writer.flush()
    summary_writer.close()

