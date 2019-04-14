from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os

config_file = "../home/ramazam/Documents/maskrcnn-benchmark/configs/seed/e2e_faster_rcnn_R_50_C4_1x_seed.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
print(coco_demo)
# load image and then run prediction

# Spine
folder = "/home/giancos/git/Spine_Dendrite_Collab/data/Labeled_Spines_Tavita/"
path_image = os.path.join(folder, "spine000001", "spine_image000001.tif")
path_results = path_image.replace(".tif", "_results.png")

# Seeds
# folder = "/home/giancos/git/SeedCounter/data"
# path_image = os.path.join(folder, "image050.jpg")
# path_results = path_image.replace(".jpg", "_results.png")


image = cv2.imread(path_image)
predictions = coco_demo.run_on_opencv_image(image)
# print(predictions)
cv2.imwrite(path_results, predictions)