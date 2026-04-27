from copy import deepcopy
from datetime import datetime
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
import yaml

import sys
sys.path.append(os.getcwd())
from embed.embed_util import load_label_plain

class TTNLogit():
    def __init__(self, args):
        # Load config and save necessary input arguments.
        self.add_config_param(args.config)
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.pl_src = args.pseudo_label_source

    def add_config_param(self, config_f, keys=[]):
        with open(config_f, "r") as f: conf = yaml.safe_load(f)
        if keys == []: keys = conf.keys()
        for k in keys: setattr(self, k, conf[k])

    def load_data(self, ref_dir, pl_dir):
        # Load reference and pseudo-label embedding patches and labels using config.

        # Load reference label patch embeddings.
        embed_dir = os.path.join(self.data_dir, "embed", self.dataset)
        ref_embed_f = os.path.join(embed_dir, f"{self.embed_f}_ref.pk")
        with open(os.path.join(embed_dir, f"{self.dataset}_fo_filelist.pk"), "rb") as f:
            filelist = pickle.load(f)
        ref_embed, ref_label, _ = load_embeddings_and_labels(ref_embed_f, ref_dir, filelist)
        classes = load_class_list(f"./dataset/{self.dataset}.yaml")
        ref_idx, _ = self.make_trainval_index(ref_label, "prune", classes, ps=False)
        ref_embed, ref_label = ref_embed[ref_idx][:,0], ref_label[ref_idx][:,0]

        # Load pseudo-label patches.
        pl_embed_f = os.path.join(embed_dir, f"{self.embed_f}_{self.pl_src}.pk")
        pl_embed, pl_label, _ = load_embeddings_and_labels(pl_embed_f, pl_dir, filelist)

        embed = {"ref": ref_embed, "pl": pl_embed}
        label = {"ref": ref_label, "pl": pl_label}

        return embed, label, classes

    def make_trainval_index(self, labels, split, classes, pl=False, ps=True):

        # Make split indices.
        n = len(labels)
        if pl: nref_per_class = self.data[split]["pl_ntrain_per_class"]
        else: nref_per_class = self.data[split]["nref_per_class"]
        train_idx = torch.tensor([]).to(torch.int)
        val_idx = torch.tensor([]).to(torch.int)
        for i in range(len(classes)):
            class_i = torch.argwhere(labels[:]==i)
            if len(class_i) < nref_per_class: n_i = len(class_i)
            else: n_i = nref_per_class
            sub_i = np.linspace(0, len(class_i)-1, n_i).astype(int)
            val_i = int(n_i * self.data[split]["val_ratio"])
            if val_i == 0:
                train_idx = torch.cat((train_idx, class_i[sub_i]))
            else:
                train_idx = torch.cat((train_idx, class_i[sub_i[:-val_i]]))
                val_idx = torch.cat((val_idx, class_i[sub_i[-val_i:]]))

        # Log data details.
        ds = []
        for name, idx in zip(["train", "val"], [train_idx, val_idx]):
            ds.append(f"{split} {name} class distribution ({len(idx)} total):")
            for i in range(len(classes)):
                n_class_labels = torch.sum(labels[idx]==i)
                ds.append(f"{classes[i]:15s} has {n_class_labels} {name} examples.") 
        if ps: 
            if al: data_log = os.path.join(self.model_dir, f"{split}_al_data.txt")
            else: data_log = os.path.join(self.model_dir, f"{split}_data.txt")
            print_to_text(data_log, ds, add_text=True, make_dir=True)
        else: 
            for s in ds: print(s)

        return train_idx, val_idx

def load_embeddings_and_labels(embed_f, label_dir, file_list):

    # Load labels (output)
    print(f"Loading labels for {label_dir} as output.")
    labels = []
    label_ct = np.zeros(len(file_list)).astype(int)
    for i, f in enumerate(tqdm(file_list)):
        img_labels = load_label_plain(os.path.join(label_dir, f))
        label_ct[i] = len(img_labels)
        for l in img_labels:
            labels.append(l[0])
    labels = torch.from_numpy(np.array(labels))

    # Load embeddings (input)
    print(f"Loading embeddings for {embed_f} as input.")
    with open(embed_f, "rb") as f:
        img_patch_embeddings = pickle.load(f)
    for i in img_patch_embeddings:
        if type(img_patch_embeddings[i]) is np.ndarray:
            embed_dim = len(img_patch_embeddings[i][0])
            break
    embeddings = torch.zeros((len(labels), embed_dim), dtype=torch.float16)
    ct = 0
    skipped_images = 0
    skipped_labels = 0
    skip_mask = torch.zeros(len(labels), dtype=bool) 
    for idx in tqdm(img_patch_embeddings.keys()):
        img_embed = img_patch_embeddings[idx]
        if type(img_embed) is np.ndarray:
            for e in img_embed:
                embeddings[ct] = torch.from_numpy(e)
                ct += 1
        # If there was an error on sample image, need to skip all labels for None.
        elif img_embed == None: 
            skip_mask[ct:ct+label_ct[idx]] = True
            ct += label_ct[idx]
            skipped_images += 1
            skipped_labels += label_ct[idx]
        else:
            print(f"Error for img_embed {img_embed}")
            IPython.embed()
    print(f"Embedding count is {ct} and label count is {len(labels)}.")
    if skipped_images > 0:
        print(f"Had to skip {skipped_images} images and {skipped_labels} " \
              "labels for None patch embeddings.")
        embeddings = embeddings[~skip_mask]
        labels = labels[~skip_mask]

    return embeddings, labels, skip_mask

def load_class_list(dataset_config):
    with open(dataset_config, "r") as f:
        ds_param = yaml.safe_load(f)
    class_name_list = [ds_param["names"][k] for k in ds_param["names"].keys()]
    return class_name_list

def print_to_text(
    file_name, 
    statements, 
    add_return=True, 
    make_dir=False, 
    add_text=False
):

    if isinstance(statements, str):
        statements = [statements]
    if make_dir:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    if add_text: output_file = open(file_name, "a")
    else: output_file = open(file_name, "w")

    for statement in statements:
        if add_return:
            statement = f"{str(statement)}\n"
        output_file.write(statement)

    output_file.close()

def inference_logits(model, emb, embr, bat_sz=500):
    # Find logits for all embeddings given the reference embeddings.
    emb, embr = emb.unsqueeze(1), embr.repeat(bat_sz,1,1)
    device = model.device
    n = len(emb)
    logits_comb = torch.zeros(n).to(device)
    for i in range(0, n, bat_sz):
        bat_emb = emb[i:i+bat_sz]
        ni = len(bat_emb)
        bat_in = torch.concatenate((embr[:ni], bat_emb),1)
        with torch.no_grad(): 
            logits, _ = model(bat_in)
        logits_comb[i:i+bat_sz] = logits[:,-1]

    return logits_comb

