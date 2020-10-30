import os

data_dir = "/mnt/sdb/public/data/jerry/FairMOT/"

mot15_val = "2DMOT2015/train"

total = 0
with open('my_mot15.val', 'w') as fout:
    for d1 in os.listdir(os.path.join(data_dir, mot15_val)):
        count = 0
        print(d1)
        for f1 in os.listdir(os.path.join(data_dir, mot15_val, d1, "img1")):
            if f1.endswith('.jpg'):
                count += 1
                # print(f1)
                fout.write(mot15_val + '/' + d1 + '/' + "img1" + '/' + f1 + '\n')
        total += count
        print('count:', count)
print('total:', total)