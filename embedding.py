import csv

import requests
import time
from tqdm import tqdm

# 全局配置
config = {
    'csv_file_path': './DataPreprocessing/small.csv',
    'encoding': 'GB2312'
}


# 1. 打开CSV文件并按行分块
def process_csv_file(csv_file_path, encoding):
    """
    打开CSV文件，读取每一行的内容并进行分块, 按行分块后返回一个包含所有分块文本的列表
    :param csv_file_path: CSV文件路径
    :return: 返回一个包含所有分块文本的列表
    """
    all_chunks = []

    # 打开CSV文件并按行读取
    with open(csv_file_path, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)  # 使用DictReader以便根据列名访问数据

        for row in reader:
            # 将行数据进行分块
            all_chunks.extend([' '.join([f"{key}: {value}" for key, value in row.items()])])

    return all_chunks


def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "quentinz/bge-large-zh-v1.5",
            "prompt": text
        }
    )
    embedding = res.json().get('embedding', None)
    if embedding is None:
        # 如果 'embedding' 不存在，打印出响应内容
        print("Error: 'embedding' not found in response")
        print("Response JSON:", res.json())
        return [0, 0, 0, 0]
    return embedding


def run(csv_file_path, encoding):
    vector_list = []
    chunk_list = process_csv_file(csv_file_path, encoding)
    for chunk in tqdm(chunk_list, desc='Embedding'):
        vector = ollama_embedding_by_api(chunk)
        vector_list.append(vector)
    return vector_list, chunk_list


# 示例调用
if __name__ == '__main__':
    # csv_file_path = './DataPreprocessing/small.csv'  # 替换为你的CSV文件路径
    # chunks = process_csv_file(csv_file_path)
    #
    # # 打印分块后的结果（只显示前几个块）
    # for i, chunk in enumerate(chunks[:999], 1):
    #     print(f"chunk{i}: [{chunk}]")
    # 记录时间

    start_time = time.time()
    run(config['csv_file_path'])
    end_time = time.time()
    print("time:", end_time - start_time)
