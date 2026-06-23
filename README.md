# Turing Test Network (TTN)

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

Implementation for paper "The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning"

![alt text](./figure/overview.jpeg?raw=true "TTN Overview")
**Zero-Shot Pseudo-Label Pruning:** A single Turing Test Network (TTN) trained strictly on image-classification (bottom) finds and rejects systemic VLM hallucinations across diverse detection datasets and pseudo-label architectures (top) while accepting accurate labels (middle). TTN rejects labels for spatial inaccuracy (**A**), semantic inconsistency (**B**), or both (**C**). Visualizations generated using [FiftyOne](https://github.com/voxel51/fiftyone).

## Using TTN

### Embedding Generation

**Input:** YOLOv5-formatted dataset and pseudo-labels.

**Output:** Reference embeddings, pseudo-label embeddings, and filelist.

[FiftyOne](https://github.com/voxel51/fiftyone) dependency to generate paper embeddings (``pip install fiftyone``). Paper detection datasets are formatted as [YOLOv5](https://docs.voxel51.com/user_guide/import_datasets.html#yolov5) (see example `./dataset/voc.yaml`).

Download [example dataset and pseudo-labels](https://www.dropbox.com/scl/fi/napaltjo2ayea1ugzvoyy/data.zip?rlkey=781beco6pw5h3pjbwlw5gibjh&st=643dmnw3&dl=0) and unzip to `./data`.

Run
```
python embed/generate_label_embedings.py --data_dir ./data --dataset voc --pseudo_label_source yoloe-11l-seg-conf30
```

Output label patch embeddings and corresponding filelist for TTN pruning will be located in `./data/embed`.

For custom dataset and pseudo-labels, replicate process or provide own `float16` CLIP `ViT-L-14` label patch embeddings for subsequent TTN Pruning.

### TTN Pruning

**Input:** Reference labels, pseudo-labels, preprocessed embeddings, and filelist.

**Output:** TTN prune logits.

[PyTorch](https://pytorch.org/get-started/locally/) dependency to run TTN model.

Download [TTN models and example reference labels, pseudo-labels, preprocessed embeddings, and filelist](https://www.dropbox.com/scl/fi/hum3q5f0uli9emnp71rp1/data.zip?rlkey=i6jj7prkmwqe3fys4o63ciifx&st=ybne7gir&dl=0) and unzip to `./data`.

Run
```
python ttn/generate_ttn_logits.py --data_dir ./data --dataset voc --pseudo_label_source yoloe-11l-seg-conf30 --prune_model ttn --config ./config/ttn_logit.yaml --device cpu
```
Recommend setting `--device` to "mps" or "cuda". Can also use paper's detection fine-tuned pruning model using `--prune_model ttnd`.

Output TTN pruning logits will be located in `./data/model/logits`. Paper implementation pruned all pseudo-labels with logit score > 0. Raising the threshold increases recall but lowers precision (e.g., > 0.1).

## Training TTN

(in progress)

### Training TTN on Image Classification Datasets

**Input:** Preprocessed image embeddings and labels.

**Output:** TTN model weights.

## Citation

If you find this code useful, please consider citing our paper (will add paper pdf link soon):

```bibtex
@inproceedings{griffin26eccv,
  author={Griffin, Brent A. and Corso, Jason J.},
  title={The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning},
  booktitle={The European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Feature Request

The paper's full TTN benchmark framework is quite extensive. If there is a feature missing that you would like to implement, please create a request (as an issue is fine) and we will address when able. Cheers!
