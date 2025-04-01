import pandas as pd
import chardet
import random

# # 检测文件编码
# with open('儿科_修正.csv', 'rb') as f:
#     result = chardet.detect(f.read())
#     encoding = result['encoding']

# 读取CSV文件
df = pd.read_csv('erke_100k.csv', encoding='GB2312')

# 确保第一行是列名
# 如果不是列名，可以使用names参数指定列名，或者使用header参数指定哪一行是列名

# 随机抽取1000条数据
sample_df = df.sample(n=1000)

# 生成新的small.csv文件
sample_df.to_csv('small.csv', index=False, encoding='GB2312')

