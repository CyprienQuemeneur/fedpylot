This README is currently under construction 🚧.



## Table of Contents

- [Introduction](#-introduction-)
- [Tutorial](#-tutorial-)
- [Citation](#️-citation-)
- [Acknowledgement](#-acknowledgement-)
- [License](#-license-)

## Introduction

**Official repository** of:
- [Cyprien Quéméneur](https://scholar.google.com/citations?hl=en&user=qQ5fKGgAAAAJ),
[Soumaya Cherkaoui](https://scholar.google.be/citations?user=fW60_n4AAAAJ). 
[**FedPylot: Navigating Federated Learning for Real-Time Object Detection in Internet of Vehicles**](https://arxiv.org/abs/2406.03611).

For questions or inquiries about this program, please contact
[cyprien.quemeneur@protonmail.com](mailto:cyprien.quemeneur@protonmail.com).

## Tutorial

Here we describe how to get started with FedPylot and reproduce the results of our paper.

### Installation

We encourage to install FedPylot both locally and on your computer cluster, as a local env will be more suited for
preparing the data and can help for prototyping.

```bash
git clone https://github.com/CyprienQuemeneur/fedpylot.git
```

To install the necessary packages in your local virtual environment run:

```bash
pip install -r requirements.txt
```

Installing all the packages on your cluster can come with some subtleties, and we would advise to refer to the
documentation of your cluster for package installation and loading.

### Downloading the datasets

We used two publicly available object detection datasets in our paper: KITTI and nuImages. 

### Data preparation

Preparing the data involve both converting the annotations to the YOLO format and splitting the samples among the
federated participants. In our experiments, we assume that the server holds a separate validation set and is 
responsible for evaluating the global model.

Data preparation should ideally be run locally. Splitting the original dataset will create a folder for each
federated participants (server and clients) which will contain the samples and labels. Archiving the folders before
sending them to the cluster is recommended and can be performed automatically by the preparation scripts. A good way
to securely and reliably transfer a large volume of data to the cloud is to use a tool such as 
[Globus](https://www.globus.org/).

#### KITTI

By default, 25% of the training data is sent to the central server, as KITTI does not
feature a predefined validation set. For the remaining data, we perform a balanced and IID split among 5 clients.
The DontCare attribute is ignored. The random seed is fixed so that splitting is reproducible. To perform both the
split and the annotation conversion, run the following:

```bash
python datasets/prepare_kitti.py --data data/kitti.yaml --tar
```

#### nuImages

nuImages feature a predefined validation set which is stored on the server, while the training data is split non-IID
among 10 clients based on the locations and timeframes at which the data samples were captured.

Run the following to create the split which retains only 10 classes based on the nuScenes competition:
```bash
python datasets/prepare_nuimages.py --data data/nuimages.yaml --class-map 10 --tar
```

And the following to retain the full long-tail distribution with 23 classes:
```bash
python datasets/prepare_nuimages.py --data data/nuimages.yaml --class-map 23 --tar
```

### Downloading pre-trained weights

Starting federated learning after a pre-training phase has been shown to reduce the gap with centralized learning, thus
we use official YOLOv7 weights pre-trained on MS COCO to initialize an experiment. Downloading the appropriate weights
is normally performed by the script that launches the job, but you need to do it manually if Internet connexions are
not available on the computing nodes of your cluster.

FedPylot supports all YOLOv7 variants. For example, to download pre-trained weights for YOLOv7-tiny run the following:

```bash
bash weights/get_weights.sh yolov7-tiny
```

The pre-trained weights will then be added to the `weights/yolov7/` directory.

### Launching a job

We provide an example job script for the centralized and the federated settings, assuming the cluster supports the
Slurm Workload Manager. To launch a federated experiment, you will need to modify `run_federated.sh` and then run
the command:

```bash
sbatch run_federated.sh
```

### Inference speed

For simplicity, speed measurements were performed on Google Colab.

## Citation
If you find FedPylot is useful in your research or applications, please consider giving us a star 🌟 and citing our
paper.

```latex
@article{fedpylot2024,
      title = {{FedPylot}: Navigating Federated Learning for Real-Time Object Detection in {Internet} of {Vehicles}}, 
      author = {Quéméneur, Cyprien and Cherkaoui, Soumaya},
      journal = {arXiv preprint arXiv:2406.03611},
      year = {2024}
}
```

## Acknowledgement
We sincerely thank the authors of [YOLOv7](https://github.com/WongKinYiu/yolov7) for providing their code to
the community!

## License
FedPylot is under the [GPL-3.0 Licence](./LICENSE).
