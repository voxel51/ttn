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

## Using TTN

(in progress)

### Embedding Generation

[FiftyOne](https://github.com/voxel51/fiftyone) dependency to generate paper embeddings (``pip install fiftyone``). Paper detection datasets are formatted as [YOLOv5](https://docs.voxel51.com/user_guide/import_datasets.html#yolov5) (see example `./dataset/voc.yaml`).

Download [example dataset and pseudo labels](https://www.dropbox.com/scl/fi/napaltjo2ayea1ugzvoyy/data.zip?rlkey=781beco6pw5h3pjbwlw5gibjh&st=643dmnw3&dl=0) and unzip to `./data`.

Run
```
python embed/generate_label_embedings.py --data_dir ./data --dataset voc --pseudo_label_source yoloe-11l-seg-conf30
```

Output label patch embeddings for TTN pruning will be located in `./data/embed`.

### TTN Pruning



## Citation

If you find this code useful, please consider citing our paper (will add link and accurate bibtex soon):

```bibtex
@article{Griffin_2026_TTN,
  author={Griffin, Brent A. and Corso, Jason J.},
  title={The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning},
  journal={arXiv preprint},
  year={2026}
}
```
