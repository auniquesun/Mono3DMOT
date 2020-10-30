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


data_dir = 'data'
# img_dir = os.path.join(data_dir, 'images')
# img_dir = os.path.join(data_dir, 'example')
img_dir = os.path.join(data_dir, '1')


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    # 2: 处理带有文件夹结构的多张图片
    for sub_img_dir in os.listdir(img_dir):
        for sub_sub_img_dir in os.listdir(os.path.join(img_dir, sub_img_dir)):
            for f_name in os.listdir(os.path.join(img_dir, sub_img_dir, sub_sub_img_dir)):
                if f_name.endswith('.jpg'):
                    begin, end = 0, f_name.find('.')
                    prefix = f_name[begin:end]
                    result_filename = os.path.join(result_root, sub_img_dir, sub_sub_img_dir, prefix + '.json')

                    # 处理图片
                    input_image = os.path.join(img_dir, sub_img_dir, sub_sub_img_dir, prefix + '.jpg')
                    dataloader = datasets.LoadImages(input_image, opt.img_size)

                    save_dir = os.path.join(result_root, sub_img_dir, sub_sub_img_dir)
                    # 评估结果
                    eval_seq(opt, dataloader, 'mot', result_filename, save_dir=save_dir, show_image=False, img_name = prefix)
    
    # 3: 处理视频
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
