#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xlwt
header = ['序号','姓名','年龄']

data = [
    {'id':1,'name':'mike','age':18},
    {'id':2,'name':'jack','age':18},
    {'id':3,'name':'lina','age':16},
    {'id':4,'name':'lnda','age':20},
]

book = xlwt.Workbook(encoding='utf-8', style_compression=0) # 创建一个Workbook对象，这就相当于创建了一个Excel文件
sheet = book.add_sheet('test', cell_overwrite_ok=True)  # # 其中的test是这张表的名字,cell_overwrite_ok，表示是否可以覆盖单元格，其实是Worksheet实例化的一个参数，默认值是False

# 设置表头
i = 0
for k in header:
    sheet.write(0, i, k)
    i = i + 1

# 数据写入excel
row = 1
for val in data:
    print(val)
    sheet.write(row, 0, val['id'])  # 第二行开始
    sheet.write(row , 1, val['name'])  # 第二行开始
    sheet.write(row , 2, val['age'])  # 第二行开始
    row = row + 1

# 最后，将以上操作保存到指定的Excel文件中
book.save(r'test1.xls')  # 在字符串前加r，声明为raw字符串，这样就不会处理其中的转义了。否则，可能会报错