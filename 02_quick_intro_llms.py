from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain


def question_answer_prompt_template_example():
    template = """Question: {question}

    Answer: """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )

    question = "What is the capital city of France?"

    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=hub_llm,
    )

    print(llm_chain.run(question))


if __name__ == "__main__":
    question_answer_prompt_template_example()
