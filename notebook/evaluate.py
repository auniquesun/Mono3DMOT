import os
import cv2
import json


"""
行人3D定位部分，单独写的这部分代码，利用KITTI数据集评估 ——
    行人 bbox、
    行人高度、
    内参矩阵的 fx和fy、
    图片分辨率

我需要清楚的是：一个 calib_xxxxx.txt，对应一张图片 img_xxxxx.png，对应多个行人==多个行人标签
"""


data_dir = '/mnt/sdb/public/data/kitti/object/training'
label_dir = os.path.join(data_dir, 'label_2')
img_dir = os.path.join(data_dir, 'image_2')
calib_dir = os.path.join(data_dir, 'calib')


def main():
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
                
                """
                results存放计算ground-truth和对应的计算结果，一张图片可能有多个行人，所以要设计成这样

                {
                    person_id: {'gt': ..., 'computed': ...}
                    ...
                    ...
                }
                注：一个文件可能包含多个Pedestrian，这里person_id就是Pedestrian所在文件的行数，从0开始
                """
                results = {}
                
                lines = f_label.readlines()
                for idx, line in enumerate(lines):
                    values = line.strip().split(' ')
                    if len(values) > 0 and values[0] == 'Pedestrian':
                        results[idx] ={}
                    
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
                        # print(label_file, '> ground-truth:', xyz[0], xyz[1], xyz[2])
                        # print(label_file, '> computed:', '{:.2f}'.format(computed_x), '{:.2f}'.format(computed_y), '{:.2f}'.format(computed_z))

    #                     results['gt'] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                        results[idx]['gt'] = xyz
                        results[idx]['computed'] = ['{:.2f}'.format(computed_x), '{:.2f}'.format(computed_y), '{:.2f}'.format(computed_z)]
                        
                if len(results) != 0:
                    result_name = label_file[begin:end] + '.json'
                    with open(os.path.join(data_dir, 'localization', result_name), 'w') as f_result:
                        json.dump(results, f_result)
                    print(result_name, 'saved!')


loc_dir = os.path.join(data_dir, 'localization')
def analyze_localization():
    with open(os.path.join(data_dir, 'localization', 'error_rate.json'), 'w') as fout:
        data = {}
        
        for f_name in os.listdir(loc_dir):
            if f_name.endswith('.json') and f_name != 'error_rate.json':  # 别忽略第二个条件
                f = os.path.join(loc_dir, f_name)
                with open(f) as fin:
                    results = json.load(fin)
                    data[f_name] = {}

                    for key in results.keys():
                        # convert string to float
                        gt = [ float(elem) for elem in results[key]['gt'] ]
                        computed = [ float(elem) for elem in results[key]['computed'] ]
                        # compute error
                        error = [ g-c for g,c in zip(gt, computed) ]
                        # compute error rate
                        error_rate = []
                        for e,g in zip(error, gt):
                            if g == 0:
                                error_rate.append(abs(e))
                            else:
                                error_rate.append(abs(e/g))
                        data[f_name][key] = error_rate
                        print(f_name, key, '-> error_rate:', error_rate)
                    
        json.dump(data, fout, indent=4)


def analyze_detection():
    # 仅对有行人的图片进行检测
    for f_name in os.listdir(loc_dir):
        if f_name.endswith('.json'):
            begin, end = 0, f_name.find('.')
            img_name = f_name[begin:end]
            os.path.join()

    
if __name__ == '__main__':
#     main()
    analyze_localization()
