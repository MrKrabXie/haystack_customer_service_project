import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever

import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever

from src.CustomDocument import CustomDocument


def setup_haystack_retriever():
    # 加载客服知识库数据
    with open('../data/customer_service_knowledge_base.json', 'r', encoding='utf-8') as f:
        knowledge_base_data = json.load(f)

    document_store = InMemoryDocumentStore()
    for item in knowledge_base_data:
        document = CustomDocument(
            content=item["answer"],
            meta={"question": item["question"]}
        )
        document_store.write_documents([document])

    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="bert-base-uncased",
        passage_embedding_model="bert-base-uncased",
        max_seq_len_query=64,
        max_seq_len_passage=256
    )
    document_store.update_embeddings(retriever)

    return retriever