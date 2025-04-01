import argparse
import embedding
import chromaDB
from prompt_work import QueryStandardizer


def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="命令行参数示例，包括默认值")

    # 添加命令行参数
    parser.add_argument('-c', '--csv_file_path', type=str, help="CSV 文件路径",
                        default='./DataPreprocessing/small.csv')  # 默认值
    parser.add_argument('-e', '--encoding', type=str, help="文件编码", default='GB2312')  # 默认值
    parser.add_argument('-d', '--db_path', type=str, help="数据库路径", default='./chromaDB.db')  # 默认值
    parser.add_argument('-n', '--collection_name', type=str, help="集合名称", default='collection_v3')  # 默认值
    parser.add_argument('--n_results', type=int, help="返回查询结果的数量", default=10)  # 默认值
    parser.add_argument('--rebuild_db', type=bool, help="是否重新构建数据库", default=False)  # 默认值

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 打印解析后的配置
    print(f"CSV 文件路径: {args.csv_file_path}")
    print(f"编码: {args.encoding}")
    print(f"数据库路径: {args.db_path}")
    print(f"集合名称: {args.collection_name}")
    print(f"返回查询结果的数量: {args.n_results}")
    print(f"是否重新构建数据库: {args.rebuild_db}")

    if args.rebuild_db:
        # 重建数据库
        embeddings, documents = embedding.run(args.csv_file_path, args.encoding)
        collection = chromaDB.run(path=args.db_path,
                                  collection_name=args.collection_name,
                                  documents=documents,
                                  embeddings=embeddings)
    else:
        # 加载数据库集合
        collection = chromaDB.test_getdb(path=args.db_path, collection_name=args.collection_name)

    standardizer = QueryStandardizer()
    while True:
        # 获取用户输入
        qs = input("请输入问题：")
        # 标准化输入
        qs = standardizer.standardize_query(qs)
        print(f"标准化后的问题: {qs}")
        # 获取查询内容的嵌入向量
        qs_embedding = embedding.ollama_embedding_by_api(qs)
        # 查询数据库
        res = collection.query(query_embeddings=[qs_embedding, ], query_texts=qs, n_results=args.n_results)

        # 打印查询结果
        print("查询结果：")
        for i in range(args.n_results):
            try:
                result = res["documents"][0][i + 1]  # 获取第i个结果
                print(f"结果 {i + 1}:")
                print('\n'.join(result.split()))
                print(
                    "============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================")
            except IndexError:
                print(f"结果 {i + 1}: 无结果")


if __name__ == '__main__':
    main()
