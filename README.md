## Introduction
Due to the low cost and ease of use of monocular cameras, it will have a broad application prospect if the 3D spatial locations of pedestrians in the surveillance area can be accurately located and tracked by monocular cameras. To address the lack of efficient and accurate end-to-end pedestrian 3D localization and tracking methods based on monocular cameras, we design and implement Mono3DMOT, a real-time multi-pedestrian detection, 3D localization and tracking method using only monocular cameras. It first performs joint learning of pedestrian detection and ID embedding representation, then solves the monocular scale ambiguity problem by assuming pedestrian height distribution. We also proposes a 3D location calculation method without additional parameters, and use target location and target embedding for pedestrians matching to improve the efficiency and accuracy of 3D localization and tracking. Comparison experiments with the current state-of-the-art methods in several public data sets show that Mono3DMOT can accurately track pedestrian 3D locations and achieve real-time operational efficiency.

## Installation
* Clone this repo, and we'll call the directory that you cloned as ${Mono3DMOT_ROOT}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
conda create -n Mono3DMOT
conda activate Mono3DMOT
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${Mono3DMOT_ROOT}
pip install -r requirements.txt
cd src/lib/models/networks/DCNv2_new sh make.sh
```
* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 
* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

We use the same training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT). Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16. 

[2DMOT15](https://motchallenge.net/data/2D_MOT_2015/) and [MOT20](https://motchallenge.net/data/MOT20/) can be downloaded from the official webpage of MOT challenge. After downloading, you should prepare the data in the following structure:
```
MOT15
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```
Then, you can change the seq_root and label_root in src/gen_labels_15.py and src/gen_labels_20.py and run:
```
cd src
python gen_labels_15.py
python gen_labels_20.py
```
to generate the labels of 2DMOT15 and MOT20. The seqinfo.ini files of 2DMOT15 can be downloaded here [[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w), [[Baidu],code:8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g).

## Pretrained models and baseline model
* **Pretrained models**

DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view).
HRNetV2 ImageNet pretrained model: [HRNetV2-W18 official](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw), [HRNetV2-W32 official](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w).
After downloading, you should put the pretrained models in the following structure:
```
${Mono3DMOT_ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
           └——————hrnetv2_w32_imagenet_pretrained.pth
           └——————hrnetv2_w18_imagenet_pretrained.pth
```
* **Baseline model**

Our baseline Mono3DMOT model can be downloaded here: DLA-34: [[Google]](https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu) [[Baidu, code: 88yn]](https://pan.baidu.com/s/1YQGulGblw_hrfvwiO6MIvA). HRNetV2_W18: [[Google]](https://drive.google.com/open?id=182EHCOSzVVopvAqAXN5o6XHX4PEyLjZT) [[Baidu, code: z4ft]](https://pan.baidu.com/s/1h1qwn8dyJmKj_nZi5H3NAQ).
After downloading, you should put the baseline model in the following structure:
```
${Mono3DMOT_ROOT}
   └——————models
           └——————all_dla34.pth
           └——————all_hrnet_v2_w18.pth
           └——————...
```

## Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Run:
```
sh experiments/all_dla34.sh
```

## 3D Localization
1. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and unpack it

2. Classify pedestrians according to distance and mode in KITTI, don't forget to save the results in `distance_and_mode.json`, referring to `notebook/3d_visualization.ipynb`

3. Predict 3D location of pedestrians in KITTI and compute errors with ground truth

4. Visualize the results using `seaborn`

## 3D Tracking
* The default settings run tracking on the validation dataset from 2DMOT15. Using the DLA-34 baseline model, you can run:
```
cd src
python track.py mot --load_model ../models/all_dla34.pth --conf_thres 0.6
```
to see the tracking results (76.1 MOTA using the DLA-34 baseline model). You can also set save_images=True in src/track.py to save the visualization results of each frame. 

Using the HRNetV2-W18 baseline model, you can run:
```
cd src
python track.py mot --load_model ../models/all_hrnet_v2_w18.pth --conf_thres 0.6 --arch hrnet_18 --reid_dim 128
```
to see the tracking results (76.6 MOTA using the HRNetV2-W18 baseline model).

* To get the txt results of the test set of MOT16 or MOT17, you can run:
```
cd src
python track.py mot --test_mot17 True --load_model ../models/all_dla34.pth --conf_thres 0.4
python track.py mot --test_mot16 True --load_model ../models/all_dla34.pth --conf_thres 0.4
```
and send the txt files to the [MOT challenge](https://motchallenge.net) evaluation server to get the results. (You can get the SOTA results 67.5 MOTA on MOT17 test set using the baseline model 'all_dla34.pth'.)

* To get the SOTA results of 2DMOT15 and MOT20, you need to finetune the baseline model on the specific dataset because our training set do not contain them. You can run:
```
sh experiments/ft_mot15_dla34.sh
sh experiments/ft_mot20_dla34.sh
```
and then run the tracking code:
```
cd src
python track.py mot --test_mot15 True --load_model your_mot15_model.pth --conf_thres 0.3
python track.py mot --test_mot20 True --load_model your_mot20_model.pth --conf_thres 0.3 --K 500
```
Results of the test set all need to be evaluated on the MOT challenge server. You can see the tracking results on the training set by setting --val_motxx True and run the tracking code. We set 'conf_thres' 0.4 for MOT16 and MOT17. We set 'conf_thres' 0.3 for 2DMOT15 and MOT20. You can also use the SOTA MOT20 pretrained model here [[Google]](https://drive.google.com/open?id=1GInbQoCtp1KHVrhzEof77Wt07fvwHcXb), [[Baidu],code:mqnz](https://pan.baidu.com/s/1c0reh3XDnbeoPZ4zFIvzGg):
```
python track.py mot --test_mot20 True --load_model ../models/mot20_dla34.pth --reid_dim 128 --conf_thres 0.3 --K 500
```
After evaluating on MOT challenge server, you can get 58.7 MOTA on MOT20 test set using the model 'mot20_dla34.pth'.

## Demo
You can input a raw video and get the demo video by running src/demo.py and get the mp4 format of the demo video:
```
cd src
python demo.py mot --load_model ../models/all_dla34.pth --conf_thres 0.4
```
You can change --input-video and --output-root to get the demos of your own videos.

If you have difficulty building DCNv2 and thus cannot use the DLA-34 baseline model, you can run the demo with the HRNetV2_w18 baseline model (don't forget to comment lines with 'dcn' in src/libs/models/model.py if you do not build DCNv2): 
```
cd src
python demo.py mot --load_model ../models/all_hrnet_v2_w18.pth --arch hrnet_18 --reid_dim 128 --conf_thres 0.4
```
--conf_thres can be set from 0.3 to 0.7 depending on your own videos.

## Reference
* [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)
* [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)
* [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)
