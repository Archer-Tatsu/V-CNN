# V-CNN
Viewport-based CNN for visual quality assessment on 360° video.

Note that this is an updated version of the approach in our [CVPR2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Viewport_Proposal_CNN_for_360deg_Video_Quality_Assessment_CVPR_2019_paper.html), and thus the results are further improved.
There are several differences between the CVPR2019 paper and this code.

Dataloader and the corresponding files for our [VQA-ODV](https://github.com/Archer-Tatsu/VQA-ODV) dataset are also provided.

At least 1 GPU is required by FlowNet2.


## Dependencies

* python3
* PyTorch == 1.0.1 (CUDA 9.0 is required for compilation of FlowNet2)
* s2cnn: https://github.com/jonas-koehler/s2cnn
* FlowNet2: https://github.com/NVIDIA/flownet2-pytorch
* numpy
* scipy
* scikit-image
* tqdm

## Binaries

The binaries including pre-trained model, as well as the list files for VQA-ODV in inference can be obtained [HERE](https://www.dropbox.com/sh/zblm9bnmc3dksti/AAC2zJB45WtAh4s9psVjKDIRa?dl=0).

Please put all these files under the log directory.

## Usage

```
python test.py --log_dir /path/to/log/directory --flownet_ckpt /path/to/flownet2/pre-trained/model [--batch_size 1] [--num_workers 4] [--test_start_frame 21] [--test_interval 45]
```
Note that this released version only supports `batch_size` of 1 in inference. The `num_workers` should be set according to the condition of the computer.

It may spend a lot of time to test on all frames for each sequence. Therefore, frame drop can be set via `test_start_frame` and `test_interval`.
The default settings are to test every 45 frames for each sequence, beginning with the 22 frame. 

## Reference
If you find this code useful in your work, please acknowledge it appropriately and cite the paper:
```
@inproceedings{Li_2019_CVPR,
author = {Li, Chen and Xu, Mai and Jiang, Lai and Zhang, Shanyi and Tao, Xiaoming},
title = {Viewport Proposal CNN for 360deg Video Quality Assessment},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {10177--10186},
month = {June},
year = {2019}
}
```