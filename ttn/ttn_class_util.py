import numpy as np
import os
import pickle
import torch
import yaml

from ttn_class_data import TTNClassDataset
from ttn_util import print_to_text

class TTNClass():
    def __init__(self, args=[]):

        # Load config.
        self.add_config_param(args.config)

        # Handle specific attributes from args.
        self.exp_name = f"ttn-{self.model['embed_model']}-{args.trial}"
        self.model["input_type"] = args.input_type
        self.machine["device"] = args.device
        self.data_dir = args.data_dir
        self.model_dir = os.path.join(args.data_dir,"model","weights",self.exp_name)
        self.model["pretrain"] = args.pretrain_model != ""
        if self.model["pretrain"]:
            self.model["weights"] = f"{os.path/dirname(self.model_dir)}" \
                f"/{args.pretrain_model}/best.pth"
            self.model["freeze_token_embd"] = args.freeze_token_embd
        if args.tkn_fc != []: self.model["tkn_fc"] = args.tkn_fc
        for attr in ["set"]: setattr(self, attr, getattr(args, attr))

        # Other CL arguments.
        model_attrs = ["nembd", "nhead", "nlayer", "nref", "nselect"]
        data_attrs = ["prob_fail", "ntrain_per_class"]
        train_attrs = ["epochs", "val_rate", "pos_weight", \
            "batch_size", "learning_rate", "weight_decay", \
            "set_size_per_epoch"]
        for attrs, attr_type in zip([model_attrs, data_attrs, train_attrs], \
                                    ["model", "data", "train"]):
            for attr in attrs:
                if getattr(args, attr) != -1:
                    getattr(self, attr_type)[attr] = getattr(args, attr)

    def make_train_dir(self, exist_ok=False):
        # Save train data config to model training directory.
        os.makedirs(self.model_dir, exist_ok=exist_ok)
        with open(os.path.join(self.model_dir, "train_data.yaml"), "w") as f:
            yaml.dump(self.data, f)
        with open(os.path.join(self.model_dir, "model.yaml"), "w") as f:
            yaml.dump(self.model, f)
        with open(os.path.join(self.model_dir, "train_setup.yaml"), "w") as f:
            yaml.dump(self.train, f)

    def add_config_param(self, config_f, keys=[]):
        with open(config_f, "r") as f: conf = yaml.safe_load(f)
        if keys == []: keys = conf.keys()
        for k in keys: setattr(self, k, conf[k])

    def load_data(self):
        # Load data for each data source.


        # Initiate training and validation sets.
        train_data = TTNClassDataset(
            self.model["nref"], 
            self.model["nselect"],
            self.data["prob_fail"],
            self.model["input_type"],
            set_sz=self.train["set_size_per_epoch"],
        )
        val_data = TTNClassDataset(
            self.model["nref"], 
            self.model["nselect"],
            self.data["prob_fail"],
            self.model["input_type"],
        )

        for s in self.set:

            # Setup.
            dd = self.data[f"set_{s}"] # Dataset details.
            print(f"Loading preprocessed {dd['dataset']} data.")
            embed_dir = f"{self.data_dir}/embed/{dd['dataset']}"
            dd["classes"] = pickle.load(open(f"{embed_dir}/classes.pk", "rb"))

            # Load embeddings and labels.
            embed_f = f"{embed_dir}/patch_embed_{self.model['embed_model']}_hl.pk"
            embed = torch.tensor(pickle.load(open(embed_f, "rb")))
            if self.model["input_type"] == "float32": embed = embed.to(dtype=torch.float32)
            label_f = f"{embed_dir}/labels.pk"
            labels = torch.tensor(pickle.load(open(label_f, "rb")))

            train_idx, val_idx = self.make_trainval_index(labels, s)

            # Add new indexed data to existing datasets.
            # memory_location = self.machine["device"]
            memory_location = "cpu"
            train_data.add_dataset(
                embed[train_idx][:,0].to(memory_location), 
                labels[train_idx][:,0].to(memory_location),  
            )
            val_data.add_dataset(
                embed[val_idx][:,0].to(memory_location),   
                labels[val_idx][:,0].to(memory_location),   
            )

        # Misc.
        self.embd_dim = len(embed[0])

        return train_data, val_data

    def make_trainval_index(self, labels, s, ps=True):
        
        # Setup.
        dd = self.data[f"set_{s}"]
        n = len(labels)
        ntrain_per_class = self.data["ntrain_per_class"]
        train_idx = torch.tensor([]).to(torch.int)
        val_idx = torch.tensor([]).to(torch.int)
        n_class = len(dd["classes"])

        # Make train / validation split.
        for i in range(n_class):
            class_i = torch.argwhere(labels[:]==i)
            if len(class_i) < ntrain_per_class: n_i = len(class_i)
            else: n_i = ntrain_per_class
            sub_i = np.linspace(0, len(class_i)-1, n_i).astype(int)

            # Make split indices by example (unseen example validation).
            val_i = int(n_i * self.data["val_ratio"])
            if val_i == 0:
                train_idx = torch.cat((train_idx, class_i[sub_i]))
            else:
                train_idx = torch.cat((train_idx, class_i[sub_i[:-val_i]]))
                val_idx = torch.cat((val_idx, class_i[sub_i[-val_i:]]))

        # Log data details.
        ds = []
        for name, idx in zip(["train", "val"], [train_idx, val_idx]):
            ds.append(f"Set {s} {dd['dataset']} {name} class distribution ({len(idx)} total):")
            for i in range(len(dd["classes"])):
                n_class_labels = torch.sum(labels[idx]==i)
                ds.append(f"{dd['classes'][i]:15s} has {n_class_labels} {name} examples.") 
        if ps: 
            data_log = f"{self.model_dir}/data_{s}_{dd['dataset']}.txt"
            print_to_text(data_log, ds, add_text=True, make_dir=True)
        else: print(ps)

        return train_idx, val_idx

