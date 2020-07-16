import xlwt

fname = '/home/chenkaixu/ComputerVision/print/output_x.txt'    # 文件夹路径

with open(fname, 'r+', encoding='utf-8') as f:

    for line in f.readlines():    # 按行读取每行
        # i = line[:-1].split(':')
        # print i
        # s = re.split(',|:',line)  # 多个分隔符
        s = line.split(':')
        # print(s = line[:-1].split(':')) # 切片去掉换行符，再以‘，’分割字符串 ，得到一个列表

# print(s[0])



book = xlwt.Workbook(encoding='utf-8', style_compression=0) # 创建一个Workbook对象，这就相当于创建了一个Excel文件
sheet = book.add_sheet('test', cell_overwrite_ok=True)  # # 其中的test是这张表的名字,cell_overwrite_ok，表示是否可以覆盖单元格，其实是Worksheet实例化的一个参数，默认值是False


# a = []
# data=open('print/output_1.txt','w')
row = 1
num1 = 1 
for num in range(len(s)):
    # num = num + 1
    a = s[num].split(',', 1)[1:]
    sheet.write(row, 0, a)
    b = s[num].split(',',1)[:1]
    sheet.write(row, 1, b)

    # while num1 < num:
    #     b = s[num1].split(',',1)[:1]
    #     sheet.write(row, 1, b)
    #     num1 = num1 + 1


    row = row + 1
    # print(a)
    # print(a,file = data)

# print(s[1].split(',',1)[:1])
# sheet.write(1, 1, s[1].split(',',1)[:1])
sheet.write(1, 0, s[0])
# 最后，将以上操作保存到指定的Excel文件中
book.save(r'test1.xls')  # 在字符串前加r，声明为raw字符串，这样就不会处理其中的转义了。否则，可能会报错
