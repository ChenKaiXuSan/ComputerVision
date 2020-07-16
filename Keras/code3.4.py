# python深度学习代码
# 电影评论分类:二分类问题

from keras.datasets import imdb

# train_labels 和 test_labels 都是0和1组成的列表，0代表负面，1代表正面
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # 仅保留数据中前10000个常出现的单词