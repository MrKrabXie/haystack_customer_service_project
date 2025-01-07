from haystack_utils import setup_haystack_retriever
from langchain_utils import setup_langchain_answer_generator


def main():
    retriever = setup_haystack_retriever()
    answer_generator = setup_langchain_answer_generator()

    user_question = "产品的保修期是多久？"
    retrieved_results = retriever.retrieve(query=user_question)
    retrieved_content = " ".join([result.content for result in retrieved_results])

    answer = answer_generator.run({"question": user_question, "retrieved_content": retrieved_content})
    print("回答:", answer)


if __name__ == "__main__":
    main()