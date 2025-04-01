import uuid
import chromadb
from tqdm import tqdm


def test_getdb(path="db/chroma_demo", collection_name="collection_v1"):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


# def run(path="db/chroma_demo", collection_name="collection_v1", documents=None, ids=None, embeddings = None ):
#
#     client = chromadb.PersistentClient(path=path)
#
#     # 创建集合
#     # if client.get_collection(collection_name):
#     #     client.delete_collection(collection_name)
#     collection = client.get_or_create_collection(name=collection_name)
#
#     # 构造数据
#     ids = [str(uuid.uuid4()) for _ in documents]
#
#     # 插入数据
#     collection.add(
#         ids=ids,
#         documents=documents,
#         embeddings=embeddings
#     )
#
#     return collection
#
#     # # 关键字搜索
#     # qs = "感冒胃疼"
#     # qs_embedding = ollama_embedding_by_api(qs)
#     #
#     # res = collection.query(query_embeddings=[qs_embedding, ], query_texts=qs, n_results=2)
#     # # print(res)
#     #
#     # result = res["documents"][0]
#     # print(result)

def run(path="db/chroma_demo", collection_name="collection_v1", documents=None, embeddings=None):
    client = chromadb.PersistentClient(path=path)
    # 创建或获取集合
    collection = client.get_or_create_collection(name=collection_name)
    # 分块插入数据
    # 分批插入防止内存占用过高
    batch_size = 500
    total_batches = (len(documents) + batch_size - 1) // batch_size  # 计算总批次

    for i in tqdm(range(total_batches), desc="Inserting into collection"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        batch_ids = [str(uuid.uuid4()) for _ in range(start_idx, end_idx)]
        batch_documents = documents[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings
        )

    return collection


if __name__ == '__main__':
    run()
