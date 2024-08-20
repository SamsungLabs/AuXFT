# AuXFT: Cross-Architecture Auxiliary Feature Space Translation for Efficient Few-Shot Personalized Object Detection, IROS 2024.
This repository contains the code to reproduce the results presented in our [paper](https://arxiv.org/abs/2407.01193).

![](https://media.github.ecodesamsung.com/user/24344/files/b6c87d3d-3f2b-4554-a03f-034e49146259)



## Prerequisites
1. Clone this repository.
2. Create a new python 3.11 environment using conda ([anaconda](https://www.anaconda.com/download), [miniconda](https://docs.anaconda.com/free/miniconda/)): ```conda create -n auxft python==3.11.5```
3. Install the dependencies using pip: ```pip install -r requirements.txt```
4. Download the evaluation datasets processed into YOLO format and place them into ```data```, if you wish to change the data paths you can modify the ```datasets/data_paths.yaml``` file:
    1. [CORe50](https://zenodo.org/records/13254883)
    2. [iCubWorld](https://zenodo.org/records/13254883)
    3. [PerSeg](https://zenodo.org/records/13254883)
    4. [POD]() (Currently not available to the public)
5. To train the architecture download the [OpenImages](https://storage.googleapis.com/openimages/web/download_v7.html) detection dataset and extract the subset used for this work using the script ```OI_subset.py```. The scrips needs to be run from an empty folder which will contain the new split, the full OpenImages and the new split need to be in the same root folder:
```
root +
     + OpenImagesRaw +
     |               + oidv6-class-descriptions.csv
     |               + oidv6-train-annotations-bbox.csv
     |               + oidv6-test-annotations-bbox.csv
     |               + validation-annotations-boox.csv
     |               + train +
     |               |       .
     |               |       .
     |               |       .
     |               + val   +
     |               |       .
     |               |       .
     |               |       .
     |               + test  +
     |               |       .
     |               |       .
     |               |       .
     + OpenImages    +
     |               + OI_subset.py
```
6. Once the script has finished copying the data, rearrange the folder structure to match that of a YOLO dataset. 
```
root +
     + train +
     |       + images +
     |       |        + im1.jpg
     |       |        + im2.jpg
     |       |        .
     |       |        .
     |       |        .
     |       |
     |       + labels +
     |       |        + im1.txt
     |       |        + im2.txt
     |       |        .
     |       |        .
     |       |        .
     |       |
     |
     + val +
     |     [same tree as train]
```
More info is available at: https://docs.ultralytics.com/datasets/.

## Training
To train our architecture with the default configuration run the following command.
```
torchrun --nproc-per-node=8 openImages_pretrain.py
```
During training the tensorboard logs are saved in ```logs/OI```.

For the evaluation we will assume that, after training, the checkpoint file ```yolo_final.pth``` has been copied from your log folder to ```ckpts``` with the name ```auxft_rerun.pth```.

To evaluate on iCubWorld and CORe50 one needs to first finetune the models for 10 epochs, using the provided script:
```
torchrun --nproc-per-node=1 finetune_core_icub.py --dataset [core50/icub]
```
After finetuning the checkpoints can be found in ```logs/finetune/[core50/icub]/yolo_final.pth```.

## Evaluation
To evaluate the performance on the PerSeg dataset of the architecture just trained run the ```train_protonet.py``` script as follows.

```
python train_protonet.py --ckpt "ckpts/auxft_rerun.pth"
```

To get a list of all arguments and their description run ```python train_protonet.py --help```


## Cite Us

If you find any of them useful for your research please consider citing us:
```
@inproceedings{barbato2024crossarchitectureauxiliaryfeaturespace,
      title={Cross-Architecture Auxiliary Feature Space Translation for Efficient Few-Shot Personalized Object Detection}, 
      author={Francesco Barbato and Umberto Michieli and Jijoong Moon and Pietro Zanuttigh and Mete Ozay},
      booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2024},
      organization={IEEE}
}
```
