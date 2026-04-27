import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
import yaml

import fiftyone as fo
from embed_util import load_yolov5_dataset, load_labels, load_fo_model

# Select labels and dataset.
batch_size = 10
num_workers = 4
embed_model = "clip"

def main(args):

    dataset = args.dataset
    data_dir = args.data_dir
    pseudo_label_source = args.pseudo_label_source

    # Load dataset and label list.
    dataset_config = os.path.join("dataset",f"{dataset}.yaml")
    with open(dataset_config, "r") as f:
        ds_param = yaml.safe_load(f)
    class_list = [ds_param["names"][k] for k in ds_param["names"].keys()]
    with open(os.path.join(ds_param["path"], ds_param["train"]), "r") as f:
        image_list = [x for x in f.read().strip().splitlines()]
    label_list = [x.split("images/")[-1].replace(".jpg",".txt") for x in image_list]

    # Load FO dataset.
    set_dir = os.path.join(data_dir, dataset)
    dataset_fo = load_yolov5_dataset(set_dir)
    fo_filepaths = dataset_fo.distinct("filepath")
    fo_ids = dataset_fo.distinct("id")

    # Generate reference filelist for embeddings.
    embed_dir = os.path.join(data_dir, "embed", dataset)
    os.makedirs(embed_dir, exist_ok=True)
    filelist_f = os.path.join(embed_dir, f"{dataset}_fo_filelist.pk")
    if not os.path.exists(filelist_f): 
        img_dir = os.path.join(dataset, "images") + "/"
        file_list = [f.split(img_dir)[-1].replace(".jpg", ".txt") \
            for f in fo_filepaths]
        pickle.dump(file_list, open(filelist_f, "wb"))
        print(f"{dataset} fo filelist saved to {filelist_f}.")

    # Add pseudo-labels to fo dataset.
    pl_dir = os.path.join(data_dir, "pseudo-label", dataset, pseudo_label_source)
    load_labels(dataset_fo, pl_dir, label_list, class_list)

    # Generate reference label (ground truth) and pseudo-label patch embeddings.
    embed_f_names = [f"patch_embed_{embed_model}_{s}" 
                    for s in ["ref", pseudo_label_source]]
    for fn, ln in zip(embed_f_names, ["ground_truth", pseudo_label_source]):

        # Only generate patch embeddings if they do not already exist.
        embed_f = os.path.join(embed_dir, fn)
        if os.path.exists(embed_f): continue

        # Generate patch ebmeddings.
        model_fo = load_fo_model(embed_model)
        patch_embeddings = dataset_fo.compute_patch_embeddings(
            model_fo, 
            ln,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Translate sample id to idx then save.
        print("Post processing patch embeddings.")
        for i, sample_id in tqdm(enumerate(fo_ids)):
            if type(patch_embeddings[sample_id]) is np.ndarray:
                patch_embeddings[i] = patch_embeddings[sample_id].astype(np.float16)
            else:
                patch_embeddings[i] = patch_embeddings[sample_id]
            del patch_embeddings[sample_id]
        pickle.dump(patch_embeddings, open(embed_f, "wb"))
        print(f"{dataset} {ln} patch embeddings saved to {embed_f}.")

    # Uncomment the following line for to confirm labels with visualization.
    # session = fo.launch_app(dataset_fo)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Label Patch Embedding Generation.")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="voc")
    parser.add_argument("--pseudo_label_source", type=str, default="yoloe-11l-seg-conf30")

    args = parser.parse_args()
    main(args)

