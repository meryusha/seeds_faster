# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from seed_predict import SeedPredict

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="/home/ramazam/Documents/maskrcnn-benchmark/configs/seed/e2e_faster_rcnn_R_50_C4_1x_seed_strat2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image-path",
        default="/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image001.jpg",
        metavar="FILE",
        help="path to image file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--strategy",
        type=int,
        default=2,
        help="Labeling strategy",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    seed_predict = SeedPredict(
        cfg,
        confidence_threshold=args.confidence_threshold,
        min_image_size=args.min_image_size,
    )

    
    # start_time = time.time()
    predictions = seed_predict.run_on_opencv_image(args.image_path)
        # print("Time: {:.2f} s / img".format(time.time() - start_time))
    print(predictions)
    return predictions
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
