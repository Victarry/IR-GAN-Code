# IR-GAN Code
Code of our work published in ACM MM 2020 [IR-GAN: Image Manipulation with Lingustic Instruction by Increment Reasoning](https://dl.acm.org/doi/10.1145/3394171.3413777)



## Setup ##

### 1. Generate `data` folder for CoDraw and i-CLEVR datasets

See [GeNeVA - Datasets - Generation Code](https://github.com/Maluuba/GeNeVA_datasets/)

### 2. Install Miniconda

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh

### 3. Create a conda environment for this repository

    conda env create -f environment.yml

### 4. Activate the environment

    conda activate irgan

### 5. Run visdom

    visdom

Training progress for all the experiments can be tracked in visdom which by default starts at `http://localhost:8097/`.

## Training the object detector and localizer ##

    python scripts/train_object_detector_localizer.py --num-classes=24 --train-hdf5=../GeNeVA_datasets/data/iCLEVR/clevr_obj_train.h5 --valid-hdf5=../GeNeVA_datasets/data/iCLEVR/clevr_obj_val.h5 --cuda-enabled  # for i-CLEVR
    python scripts/train_object_detector_localizer.py --num-classes=58 --train-hdf5=../GeNeVA_datasets/data/CoDraw/codraw_obj_train.h5 --valid-hdf5=../GeNeVA_datasets/data/CoDraw/codraw_obj_val.h5 --cuda-enabled  # for CoDraw

Note: The above commands also have several options, which can be found in the python script, that need to be set. Batch size (`--batch-size`) is not per-GPU but combined across GPUs.

This trains the object detector and localizer model used for evaluating GeNeVA-GAN on Precision, Recall, F1-Score, and `rsim` metrics. For comparison with results in our paper, you should skip training the model yourself and download the pre-trained models (`iclevr_inception_best_checkpoint.pth` and `codraw_inception_best_checkpoint.pth`) from the [GeNeVA Project Page](https://www.microsoft.com/en-us/research/project/generative-neural-visual-artist-geneva/).

## Training on CoDraw ##

Modify `geneva/config.yml` and @args/irgan-iclevr.args if needed and run:

    python train.py @args/irgan-iclevr.args

When training for multiple times, remember to change to exp_name and results_paths in args file.
## Training on i-CLEVR ##

Modify `geneva/config.yml` and `args/irgan-codraw.args` if needed and run:

    python train.py @args/irgan-codraw.args

## Evaluating a trained model on CoDraw test set ##

You will have to add the line `--load_snapshot=</path/to/trained/model>` to `args/irgan-codraw.args` to specify the checkpoint to load from and then run:

    python test.py @args/irgan-codraw.args 

## Evaluating a trained model on i-CLEVR test set ##

You will have to add the line `--load_snapshot=</path/to/trained/model>` to `args/irgan-iclevr.args` to specify the checkpoint to load from and then run:

    python test.py @args/irgan-iclevr.args

## Reference ##

Zhenhuan Liu, Jincan Deng, Liang Li, Shaofei Cai, Qianqian Xu, Shuhui Wang, Qingming Huang. 2020. IR-GAN: Image Manipulation with Linguistic Instruction by Increment Reasoning. In Proceedings of the 28th ACM International Conference on Multimedia (MM’20)

```bibtex
@InProceedings{Liu_2020_ACMMM,
    author    = {Zhenhuan Liu, Jincan Deng, Liang Li, Shaofei Cai, Qianqian Xu, Shuhui Wang, Qingming Huang.},
    title     = {IR-GAN: Image Manipulation with Linguistic Instruction by Increment Reasoning},
    booktitle = {In Proceedings of the 28th ACM International Conference on Multimedia (MM’20),
    month     = {Oct},
    year      = {2020}
}
```


## Acknowledgements

Our code is inspired by [GeNeVA](https://github.com/Maluuba/GeNeVA).

