import os
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def setup_langchain_answer_generator():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token"

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0, "max_length": 256}
    )
    prompt_template = PromptTemplate(
        input_variables=["question", "retrieved_content"],
        template="根据检索到的内容：{retrieved_content}，作为客服回答用户问题：{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain