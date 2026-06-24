import argparse
import numpy as np
import os
import shutil
from time import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ttn_class_util import TTNClass
from ttn_model import TTNModel
from ttn_util import plot_train_val_ddg, print_to_text

def main(args):

    # Initial setup.
    ddg = TTNClass(args)
    ddg.make_train_dir(exist_ok=args.overwrite)
    device = ddg.machine["device"]

    # Create dataloaders.
    train_data, val_data = ddg.load_data()
    #train_data[0] # Debugging
    trainloader = DataLoader(
        train_data, 
        batch_size=ddg.train["batch_size"], 
        shuffle=True, 
        pin_memory=device=="cuda", 
        prefetch_factor=ddg.machine["prefetch_factor"],
        num_workers=ddg.machine["num_w"],
    )
    testloader = DataLoader(
        val_data, 
        batch_size=ddg.train["batch_size"],
        shuffle=False, 
        pin_memory=device=="cuda", 
        prefetch_factor=4,
        num_workers=ddg.machine["num_w"],
    )

    # Create model.
    print(f"Creating torch model.")
    nembd0, nref, nlayer = ddg.embd_dim, ddg.model["nref"], ddg.model["nlayer"], 
    nhead, nembd, tkn_fc = ddg.model["nhead"], ddg.model["nembd"], ddg.model["tkn_fc"], 
    msk_attn, dropout = ddg.model["msk_attn"], ddg.model["dropout"]
    nseq = ddg.model["nselect"] + nref
    model = TTNModel(nembd0, nseq, nlayer, nhead, nembd, tkn_fc, msk_attn, dropout, device)
    if ddg.model["pretrain"]:
        model.load_state_dict(torch.load(
            ddg.model["weights"], 
            weights_only=True, 
            map_location=torch.device("cpu")
        ))
        if ddg.model["freeze_token_embd"]: # Freeze tokenizer in pre-trained model.
            for p in model.token_embedding_table.parameters(): 
                p.requires_grad = False
            if ddg.model["freeze_token_dropout"]:
                model.token_embedding_table[3].p = 0 # Set dropout to 0.
    m = model.to(device)
    print(f"Model has {sum(p.numel() for p in m.parameters())/1e6} M parameters.")

    # Train setup.
    print(f"Training model.")
    optimizer = torch.optim.SGD(
        m.parameters(), 
        lr=ddg.train["learning_rate"], 
        weight_decay=ddg.train["weight_decay"],
        momentum=0.9,
        nesterov=True,
    )
    val_rate = ddg.train["val_rate"]
    n_pass = len(trainloader) # * val_rate
    total_steps = n_pass * ddg.train["epochs"]
    n_val_pass = len(testloader)
    val_classes = [f"{ddg.data[f'set_{i}']['dataset']}" for i in args.set]

    # Log setup.
    model_dir = ddg.model_dir
    log_f = os.path.join(model_dir, "log.txt")
    val_f = os.path.join(model_dir, "val.txt")
    print_to_text(log_f, model_dir, make_dir=True)
    shutil.copyfile(args.config, os.path.join(model_dir, "config.yaml"))
    plot_f = log_f.replace(".txt", ".png")
    log_data = {"epoch":[], "loss":[], "val":[], "g_acc":[], "d_acc":[], \
                "lig_loss":[], "lig_val": [], "plot_f":plot_f}
    epoch_log = []
    best_val_acc = -0.1

    # Train model.
    if ddg.model["input_type"] == "float16": m.half() # float16
    tic = time()
    for epoch in range(1, ddg.train["epochs"]+1):
        m.train()
        running_loss = 0.0
        #for inputs, targets, _, _ in tqdm(trainloader):
        for inputs, targets, _, _ in trainloader:
            logits, loss = m(
                inputs.to(device), 
                targets.to(device), 
                pos_weight=ddg.train["pos_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        train_loss = running_loss/n_pass

        # Validation
        if epoch % val_rate == 0:
            m.eval()
            print(f"Finding validation performance for epoch {epoch}.")
            goose_acc = {classname: 0 for classname in val_classes}
            goose_freq = {classname: 0 for classname in val_classes}
            duck_acc = {classname: 0 for classname in val_classes}
            duck_freq = {classname: 0 for classname in val_classes}
            val_loss = 0.0
            seq_acc = torch.zeros(nseq)
            with torch.no_grad():
                for inputs, targets, _, set_labels in tqdm(testloader):

                    logits, loss = m(
                        inputs.to(device), 
                        targets.to(device), 
                        pos_weight=ddg.train["pos_weight"],
                    )
                    val_loss += loss

                    accuracy = abs((logits<0).type(torch.int) - targets.to(device))
                    seq_acc += torch.mean(accuracy, 0).to("cpu")

                    goose_idx = targets[:,nref:]==1
                    g_acc = accuracy[:,nref:][goose_idx]
                    g_classes = set_labels[:,nref:][goose_idx]
                    duck_idx = ~goose_idx
                    d_classes = set_labels[:,nref:][duck_idx]
                    d_acc = accuracy[:,nref:][duck_idx] 

                    for i, class_i in enumerate(val_classes):
                        gi_idx = g_classes == i
                        goose_acc[class_i] += torch.sum(g_acc[gi_idx]).item()
                        goose_freq[class_i] += torch.sum(gi_idx).item()
                        di_idx = d_classes == i
                        duck_acc[class_i] += torch.sum(d_acc[di_idx]).item()
                        duck_freq[class_i] += torch.sum(di_idx).item()
            
            # Find per-class and overall validation accuracy.
            val_str, total_gacc, total_dacc, total_acc = [], [], [], []
            for class_i in val_classes:
                if goose_freq[class_i] == 0: g_acc = 0
                else: g_acc = 100.0 * (goose_acc[class_i] / goose_freq[class_i])
                if duck_freq[class_i] == 0: d_acc = 0
                else: d_acc = 100.0 * (duck_acc[class_i] / duck_freq[class_i])
                acc = (g_acc + d_acc)/2
                val_str.append(f"Val. Acc. {class_i:35s} is {acc:.1f}% "\
                    f"(IC {d_acc:.1f}%, OOC {g_acc:.1f}%)")
                total_gacc.append(g_acc)
                total_dacc.append(d_acc)
                total_acc.append(acc)
            seq_acc /= n_val_pass
            seq_acc = [f"{se.item():.3f}" for se in seq_acc]
            val_str.insert(0, f"Val. Seq. Acc. {seq_acc} %")
            val_gacc = np.mean(total_gacc)
            val_dacc = np.mean(total_dacc)
            val_acc = np.mean(total_acc)
            val_loss = val_loss/n_val_pass
            val_loss_ratio = val_loss / train_loss
            toc = time()
            val_str.insert(0, f"[Epoch {epoch:4}] Loss: {train_loss} " \
                f"(Val {val_loss}, {val_loss_ratio:.3f} total / " \
                f"ValAcc. {val_acc:.3f}% (IC {val_dacc:.1f}%, " \
                f"OOC {val_gacc:.1f}%), {toc-tic:.3f}s.")

            # Update model weights if validation accuracy is best.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epoch_log.append(val_str[0])
                torch.save(m.state_dict(), f"{model_dir}/best.pth")
                print(f"Model saved to {model_dir}.")
                print_to_text(val_f, val_str, add_text=True)

            # Log results.
            for l in epoch_log: print(l)
            for l in val_str: print(l)
            print_to_text(log_f, val_str[0], add_text=True)
            log_data["epoch"].append(epoch)
            log_data["loss"].append(train_loss)
            log_data["val"].append(val_loss.to("cpu"))
            log_data["g_acc"].append(val_gacc)
            log_data["d_acc"].append(val_dacc)
            plot_train_val_ddg(log_data)

        elif epoch % 10 == 0:
            train_str = f"[Epoch {epoch:4}] Loss: {train_loss}, "\
                        f"{time()-tic:.3f}s."
            print_to_text(log_f, train_str, add_text=True)

    torch.save(m.state_dict(), f"{model_dir}/last.pth")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Turing Test Network (TTN) Training")

    parser.add_argument("--config", type=str, default="./config/ttn_class_model")
    parser.add_argument("--set", nargs="+", required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--pretrain_model", type=str, default="")
    parser.add_argument("--freeze_token_embd", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")

    # Various ablations.
    parser.add_argument("--nembd", type=int, default=-1)
    parser.add_argument("--tkn_fc", nargs="*", type=int, default=[])
    parser.add_argument("--nhead", type=int, default=-1)
    parser.add_argument("--nlayer", type=int, default=-1)
    parser.add_argument("--nref", type=int, default=-1)
    parser.add_argument("--nselect", type=int, default=-1)
    parser.add_argument("--ntrain_per_class", type=int, default=-1)
    parser.add_argument("--prob_fail", type=float, default=-1)
    parser.add_argument("--pos_weight", type=float, default=-1)
    parser.add_argument("--learning_rate", type=float, default=-1)
    parser.add_argument("--weight_decay", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--set_size_per_epoch", type=int, default=-1)
    parser.add_argument("--val_rate", type=int, default=-1)
    parser.add_argument("--input_type", type=str, default="float16")

    args = parser.parse_args()
    main(args)

