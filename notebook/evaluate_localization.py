import os
import cv2
import json


"""
我需要清楚的是：一个 calib_xxxxx.txt，对应一张图片 img_xxxxx.png，对应多个行人==多个行人标签
"""


data_dir = '/mnt/sdb/public/data/kitti/object/training'
label_dir = os.path.join(data_dir, 'label_2')
img_dir = os.path.join(data_dir, 'image_2')
calib_dir = os.path.join(data_dir, 'calib')

# count = 0
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
#         count += 1

        begin, end = 0, label_file.find('.')
        img_name = label_file[begin:end] + '.png'
        
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_height, img_width, _ = img.shape
        
        img_x_center, img_y_center = img_width/2, img_height/2
        

#         print('calib_dir:', calib_dir)
#         print('calib_file:', calib_file)
        with open(os.path.join(label_dir, label_file)) as f_label, open(os.path.join(calib_dir, label_file)) as f_calib:
            fx = None
            fy = None
            lines = f_calib.readlines()
            for line in lines:
                if "P2:" in line:
                    values = line.strip().split(' ')
                    if len(values) > 0:
                        fx = float(values[1])
                        fy = float(values[6])
                    break
            
            
            ltrb = None
            H = None
            xyz = None
            # 存放计算ground-truth和对应的计算结果
            results = {}
            
            lines = f_label.readlines()
            for line in lines:
                values = line.strip().split(' ')
                if len(values) > 0 and values[0] == 'Pedestrian':
                    # bbox of pedestrian in image plane
                    ltrb = values[4:8]
                    bbox_x_center = (float(ltrb[0]) + float(ltrb[2]))/2
                    bbox_y_center = (float(ltrb[1]) + float(ltrb[3]))/2
                    h = float(ltrb[3]) - float(ltrb[1])
                    
                    # pedestrian height
                    H = float(values[8])
                    # ground-truth pedestrian position in camera coordiante
                    xyz = values[11:14]
                    
                    computed_x = fy/fx * H * (bbox_x_center-img_x_center) / h
                    computed_y = H * (bbox_y_center-img_y_center) / h
                    computed_z = H * fy / h
                    print(label_file, '> ground-truth:', xyz[0], xyz[1], xyz[2])
                    print(label_file, '> computed:', '{:.2f}'.format(computed_x), '{:.2f}'.format(computed_y), '{:.2f}'.format(computed_z))
                    
#                     results['gt'] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                    results['gt'] = xyz
                    results['computed'] = ['{:.2f}'.format(computed_x), '{:.2f}'.format(computed_y), '{:.2f}'.format(computed_z)]
                    
            if len(results) != 0:
                result_name = label_file[begin:end] + '.json'
                with open(os.path.join(data_dir, 'localization', result_name), 'w') as f_result:
                    json.dump(results, f_result)