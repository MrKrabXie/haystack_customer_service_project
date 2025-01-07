import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever

# 加载测试数据
with open('../data/haystack_customer_service_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 创建文档存储
document_store = InMemoryDocumentStore()

# 将测试数据添加到文档存储中
for item in test_data:
    document = {
        "content": item["answer"],
        "meta": {"question": item["question"]}
    }
    document_store.write_documents([document])

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="bert-base-uncased",
    passage_embedding_model="bert-base-uncased",
    max_seq_len_query=64,
    max_seq_len_passage=256
)
document_store.update_embeddings(retriever)

user_question = "产品的保修期是多久？"
results = retriever.retrieve(query=user_question)
for result in results:
    print(f"问题: {result.meta['question']}")
    print(f"答案: {result.content}")