import uuid
import chromadb


def test_getdb(path="db/chroma_demo", collection_name="collection_v1"):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def run(path="db/chroma_demo", collection_name="collection_v1", documents=None, ids=None, embeddings = None ):

    client = chromadb.PersistentClient(path=path)

    # 创建集合
    # if client.get_collection(collection_name):
    #     client.delete_collection(collection_name)
    collection = client.get_or_create_collection(name=collection_name)

    # 构造数据
    ids = [str(uuid.uuid4()) for _ in documents]

    # 插入数据
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    return collection

    # # 关键字搜索
    # qs = "感冒胃疼"
    # qs_embedding = ollama_embedding_by_api(qs)
    #
    # res = collection.query(query_embeddings=[qs_embedding, ], query_texts=qs, n_results=2)
    # # print(res)
    #
    # result = res["documents"][0]
    # print(result)


if __name__ == '__main__':
    run()
