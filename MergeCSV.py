import os
import csv

#先写入columns_name
mergefile = open('./Merged.csv', 'w', newline='')
mergefile_writer = csv.writer(mergefile)
mergefile_writer.writerow(['path','x1', 'y1', 'x2', 'y2', 'label'])   
mergefile.close()

# 读取背景路径下文件列表
CSV_path = './csv/'
csv_files = os.listdir(CSV_path)

for i in csv_files:
    csvfile = open('./csv/'+ i, 'r').read()
    csvfile = csvfile[23:] #去掉第一行
    print('正在合并' + str(i) + '...')
    with open('./Merged.csv', 'a') as f:
            f.write(csvfile)
print('合并完毕')