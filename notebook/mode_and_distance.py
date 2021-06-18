import os
import cv2
import json
import math


"""
行人3D定位部分，不涉及到 FairMOT 的代码，而是单独写的这部分代码，直接利用KITTI数据集 ——
    行人 bbox、
    行人高度、
    内参矩阵的 fx和fy、
    图片分辨率

我需要清楚的是：一个 calib_xxxxx.txt，对应一张图片 img_xxxxx.png，对应多个行人==多个行人标签
"""


data_dir = '/mnt/sdb/public/data/kitti/object/training'
location_dir = os.path.join(data_dir, 'localization')

mode_and_distance = {
    'easy':{
        'd00_10': [],
        'd10_20': [],
        'd20_30': [],
        'd30_40': [],
        'd40_50': [],
        'd_gt_50': []
    },
    'moderate':{
        'd00_10': [],
        'd10_20': [],
        'd20_30': [],
        'd30_40': [],
        'd40_50': [],
        'd_gt_50': []
    },
    'hard':{
        'd00_10': [],
        'd10_20': [],
        'd20_30': [],
        'd30_40': [],
        'd40_50': [],
        'd_gt_50': []
    }
}


"""
主要用到 localization 里面的json文件
按照设置好的mode和interval，划分他们到字典 mode_and_distance
"""
import argparse

def main(args):
    for f_loc in os.listdir(location_dir):
        # 004569.json 是对应一张特殊的图片: 人物框宽度不到 1 pixel，定位误差很大，暂不考虑它
        if f_loc != '004569.json' and f_loc != 'error_rate.json' and f_loc.endswith('.json'):
            with open(os.path.join(location_dir, f_loc)) as fin:
                data = json.load(fin)
                for idx in data.keys():
                    mode = data[idx]['mode']
                    interval = data[idx]['interval']
                    if args.flag == 0:
                        mode_and_distance[mode][interval].append({'gt': data[idx]['gt'], 'computed': data[idx]['computed']})
                    else:
                        mode_and_distance[mode][interval].append({'gt': data[idx]['gt'], 'computed_normal': data[idx]['computed_normal']})
    
    # 保存中间结果
    if args.flag == 0:
        with open('mode_and_distance.json', 'w') as fout:
            json.dump(mode_and_distance, fout, indent=4)
    else:
        with open('mode_and_distance_normal.json', 'w') as fout:
            json.dump(mode_and_distance, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int, default=0, help='flag to decide mode_and_distance or mode_and_distance_normal')
    args = parser.parse_args()
    
    main(args)
