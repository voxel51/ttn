from copy import deepcopy
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

class TTNClassDataset(Dataset):
    def __init__(
            self, 
            nref=10, 
            nselect=1, 
            prob_fail=0.5, 
            input_type="float32",
            set_sz=0
        ):

        self.nref, self.nselect, self.nseq = nref, nselect, nref+nselect
        self.prob_fail = prob_fail
        self.set_sz = set_sz
        if input_type == "float32":
            self.in_type = torch.float32
        elif input_type == "float16":
            self.in_type = torch.float16

        self.hl_emb, self.hl_idx, self.hl_class = {}, {}, {}
        self.valid_idx, self.compare_classes = {}, {}
        self.class_sz = {}
        self.set_idx = []

    def add_dataset(
            self,
            hl_embed, 
            hl_labels, 
        ):

        # Embedding setup.
        s = len(self.set_idx)
        self.set_idx.append(s)
        self.ns = s+1
        self.hl_emb[s] = hl_embed

        # Label setup.
        self.hl_class[s] = hl_labels.type(torch.int)
        classes = torch.unique(self.hl_class[s])
        self.hl_idx[s], self.class_sz[s] = {}, {}
        for i in classes: 
            self.hl_idx[s][i.item()] = list(torch.argwhere(self.hl_class[s] == i)[:,0])
            self.class_sz[s][i.item()] = len(self.hl_idx[s][i.item()])

        # Examples only valid if there are sufficient reference examples.
        self.valid_idx[s] = []
        for i in classes:
            if self.class_sz[s][i.item()] >= self.nseq: 
                self.valid_idx[s] += self.hl_idx[s][i.item()]
        self.valid_len = [len(self.valid_idx[v]) for v in self.valid_idx]

        # Compare classes only valid if there are sufficient candidate examples.
        valid_cand = [v >= self.nselect for k,v in self.class_sz[s].items()]
        cand_classes = classes[valid_cand]
        self.compare_classes[s] = {}
        for i in classes: 
            self.compare_classes[s][i.item()] = cand_classes[cand_classes != i]

        # Misc. get item setup.
        self.emb_dim = len(hl_embed[0]) 
        self.device = hl_embed.device

        if self.set_sz != 0: self.length = self.ns * self.set_sz
        else: 
            self.s_idx = np.array([sum(self.valid_len[:i+1]) for i in range(self.ns)])
            self.length = self.s_idx[-1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx0):

        # Determine example and class.
        if self.set_sz != 0: 
            s = idx0 % self.ns # Even distribution across sets.
            idx = random.choice(self.valid_idx[s]).item() 
        else:
            s = sum(idx0 >= self.s_idx).item()
            if s > 0: idx0 -= self.s_idx[s-1]
            idx = self.valid_idx[s][idx0]
        class_label = self.hl_class[s][idx].item()

        # Figure out how many references will be actual hl in class labels.
        pass_mask = torch.rand(self.nseq)>self.prob_fail
        pass_mask[:self.nref] = True
        hl_ooc_mask = ~pass_mask

        # Determine out of class examples.
        ooc_class = random.choice(self.compare_classes[s][class_label]).item()

        # Select embeddings for each case.
        hl_idx = [i.item() for i in random.sample(self.hl_idx[s][class_label], pass_mask.sum())]
        hl_ooc_idx = [i.item() 
            for i in random.choices(self.hl_idx[s][ooc_class], k=hl_ooc_mask.sum())]

        # Create sequence embedding.
        emb_seq = torch.empty((self.nseq, self.emb_dim), dtype=self.in_type)
        emb_seq[pass_mask] = self.hl_emb[s][hl_idx]
        emb_seq[hl_ooc_mask] = self.hl_emb[s][hl_ooc_idx]
            
        # Create hl and class label.
        label_seq = torch.zeros(self.nseq, dtype=self.in_type)
        label_seq[~pass_mask] = 1

        # Misc.
        class_seq = self.hl_class[s][idx].repeat(self.nseq)
        set_seq = torch.tensor(s, dtype=self.in_type).repeat(self.nseq)

        return emb_seq, label_seq, class_seq, set_seq

