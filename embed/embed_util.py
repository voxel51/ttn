import numpy as np
import os
from tqdm import tqdm

import fiftyone as fo
import fiftyone.zoo as foz

def load_fo_model(embed_model):
    # Load fiftyone embedding model.
    if embed_model == "clip":
        model_fo = foz.load_zoo_model(
            "open-clip-torch", 
            clip_model="ViT-L-14", 
            pretrained="openai",
        )
    else: 
        if embed_model == "dinov2": fo_model_id = "dinov2-vitb14-torch"
        elif embed_model == "resn18": fo_model_id = "resnet18-imagenet-torch"
        model_fo = foz.load_zoo_model(fo_model_id)

    return model_fo

def load_yolov5_dataset(dataset_dir, split="train"):
    dataset_type = fo.types.YOLOv5Dataset
    dataset_fo = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        split=split,
    )
    fo.core.metadata.compute_metadata(dataset_fo)

    return dataset_fo

def load_labels(dataset_fo, label_dir, label_list, class_list):
    model_name = os.path.basename(label_dir)
    print(f"Add {model_name} labels to fo dataset.")
    fo_filepaths = dataset_fo.distinct("filepath")
    img_dir = fo_filepaths[0].split("/images/")[0] + "/images"
    fo_predict = [fo.Detections(detections=[]) for f in fo_filepaths]
    for lb_f in tqdm(label_list):

        # Setup.
        lb = load_label(os.path.join(label_dir, lb_f), x1y1whn=True, conf=False)
        if len(lb) == 0: continue
        fo_f = os.path.join(img_dir, lb_f).replace(".txt", ".jpg")
        fo_idx = fo_filepaths.index(fo_f)

        # Add labels to fo dataset.
        for l in lb:
            fo_predict[fo_idx]["detections"].append(
                fo.Detection(
                label=class_list[int(l[-1])],
                bounding_box=l[:4],
                confidence=l[-2]
            ))
    dataset_fo.set_values(model_name, fo_predict)

def load_label(label_file, x1y1whn=False, conf=False):
    lb_out = np.ones(shape=(0,6), dtype=np.float32)
    lb = load_label_plain(label_file)
    n_labels = len(lb)
    if n_labels > 0: 
        lb_out = np.ones(shape=(n_labels, 6), dtype=np.float32)
        lb_out[:,-1] = lb[:,0] # class idx

        # Does initial label include confidence score or not.
        if conf: 
            lb_out[:,-2] = lb[:,1] # confidence
            bbox = lb[:,2:] 
        else: 
            bbox = lb[:,1:]

        # Format bounding box.
        if x1y1whn: # Convert xcycwhn to x1y1whn
            # x1, y1 from xc, yc, w/2, h/2
            lb_out[:,[0,1]] = bbox[:,[0,1]] - bbox[:,[2,3]]/2 
            lb_out[:,[2,3]] = bbox[:,2:]
        else: # Convert xcycwhn to xyxyn
            # x1, y1 from xc, yc, w/2, h/2
            lb_out[:,[0,1]] = bbox[:,[0,1]] - bbox[:,[2,3]]/2 
            # x2, y2 from x1, y1, w, h
            lb_out[:,[2,3]] = lb_out[:,[0,1]] + bbox[:,[2,3]] 

    return lb_out

def load_label_plain(label_file):
    lb = []
    if os.path.isfile(label_file):
        with open(label_file, encoding="utf-8") as f:
            lb = np.array(
                [x.split() for x in f.read().strip().splitlines() if len(x)],
                dtype=np.float32
            )
    return lb

