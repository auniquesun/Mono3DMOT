from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track_test import eval_seq


# kitti_dir = '/mnt/sdb/public/data/kitti/object/training'
# img_dir = os.path.join(kitti_dir, 'image_2')
# loc_dir = os.path.join(kitti_dir, 'localization')

data_dir = 'data'
img_dir = os.path.join(data_dir, 'images')
loc_dir = os.path.join(kitti_dir, 'localization')

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    for f_name in os.listdir(loc_dir):
        if f_name.endswith('.json'):
            begin, end = 0, f_name.find('.')
            prefix = f_name[begin:end]
            result_filename = os.path.join(result_root, prefix + '.json')

    # 检测结果
    # result_filename = os.path.join(result_root, 'detection_results.json')

            # 处理图片
            input_image = os.path.join(img_dir, prefix + '.png')            
            dataloader = datasets.LoadImages(input_image, opt.img_size)
            # 评估结果
            eval_seq(opt, dataloader, 'mot', result_filename, show_image=False)
    
    # 处理视频
    # dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    # 定位结果
    # result_filename = os.path.join(result_root, 'localization_results.json')
    # frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    # eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b:v 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
