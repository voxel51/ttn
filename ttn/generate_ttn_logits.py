import argparse
import IPython
import numpy as np
import os
import pickle
from time import time
import torch
from tqdm import tqdm
import yaml

from ttn_model import load_ttn_model
from ttn_util import TTNLogit, inference_logits, print_to_text

def main(args):

    # Main Setup.
    data_dir = args.data_dir
    dataset = args.dataset
    device = args.device
    ddg = TTNLogit(args)

    # Set output directory and check if pruning already exists.
    out_dir = os.path.join(data_dir, "model", "logits", dataset, \
        ddg.pl_src, args.prune_model)
    nref_pc = ddg.data["prune"]["nref_per_class"]
    nm = int(nref_pc * ddg.data["prune"]["val_ratio"])
    nk = nref_pc - nm
    logit_f = f"{out_dir}/ttn-{nk}r.pk"
    if os.path.isfile(logit_f):
        print(f"Logits for {logit_f} already exists. Skipping pruning.")
        return

    # Load reference and pseudo-label embedding patches and labels using config.
    ref_dir = os.path.join(data_dir, dataset, "labels") 
    pl_dir = os.path.join(data_dir, "pseudo-label", dataset, ddg.pl_src)
    emb, label, classes = ddg.load_data(ref_dir, pl_dir)
    emb["ref"] = emb["ref"].to(device)

    # Load model.
    model_dir = os.path.join(data_dir, "model", "weights", args.prune_model) 
    m = load_ttn_model(model_dir, len(emb["ref"][0]), device)
    if m.input_type != "float16":
        print("Converting embeddings to float32.")
        for s in ["ref", "pl"]: emb[s] = emb[s].to(dtype=torch.float32)

    # Use all available reference and candidate slots for reference save one.
    nref = m.nseq - 1

    # Pruning setup.
    os.makedirs(out_dir, exist_ok=True)
    log_f = os.path.join(out_dir, "logit_gen_log.txt")
    sd = {"pl": np.zeros(len(label["pl"])).astype(np.float16),
          "no_sort_class": []
    }
    n_class = len(classes)

    # Pseudo-label pruning.
    tic = time()
    for ci in range(n_class):
        ps = []

        # Use class-appropriate reference and threshold patches from hl examples.
        ref_idx = np.argwhere(label["ref"] == ci)[0]
        nk = len(ref_idx)

        # Run inference on selected pl patches corresponding to single class.
        nref_iter = int(nk / nref)
        pl_class_i = np.argwhere(label["pl"] == ci)[0]
        n_cand = len(pl_class_i)
        pl_score = torch.zeros(n_cand)
        ps.append(f"\n[{ci}/{n_class}] Finding {classes[ci]} prune " \
                  f"score for {nk} ref and {n_cand} candidate labels.")
        if n_cand == 0: 
            print_to_text(log_f, ps, add_text=True); continue
        if nk < nref:
            ps.append("Skipping class with only {nk} reference.")
            sd["no_sort_class"].append(ci)
            print_to_text(log_f, ps, add_text=True); continue
        print(f"Processing {ddg.pl_src} {classes[ci]:15s} [{ci+1:2}/{n_class}]" \
              f" {time() - tic:05.2f}s.")
        pl_emb_i = emb["pl"][pl_class_i]

        # Iterate over reference embeddings.
        for k in tqdm(range(0, nk, nref)): 
            k_i = ref_idx[k:k+nref] 
            if len(k_i) < nref: 
                k_i = torch.concatenate((k_i, ref_idx[:nref-len(k_i)]))
            logit_k = inference_logits(m, pl_emb_i.to(device), emb["ref"][k_i])
            pl_score += logit_k.to("cpu")
        pl_score /= nref_iter
        sd["pl"][pl_class_i] = pl_score / nref_iter

        # Log summary.
        toc = time()
        ps.append(f"Prune pl score: mean {torch.mean(pl_score)}, med " \
                  f"{torch.median(pl_score)}, min {torch.min(pl_score)}, max " \
                  f"{torch.max(pl_score)}, time: {toc - tic}s.")
        print_to_text(log_f, ps, add_text=True)

    # Save prune scores.
    pickle.dump(sd, open(logit_f, "wb"))
    print(f"Logit prune results saved to {logit_f}.")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Turing Test Network Logit Generation.")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="voc")
    parser.add_argument("--pseudo_label_source", type=str, default="yoloe-11l-seg-conf30")
    parser.add_argument("--prune_model", type=str, default="ttn")
    parser.add_argument("--config", type=str, default="./config/ttn_logit.yaml")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(args)

