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


def multiple_questions_example():
    template = """Question: {question}

        Answer: """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )

    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=hub_llm,
    )

    qa = [
        {"question": "What is the capital city of France?"},
        {"question": "What is the largest mammal on Earth?"},
        {"question": "Which gas is most abundant in Earth's atmosphere?"},
        {"question": "What color is a ripe banana?"},
    ]

    res = llm_chain.generate(qa)

    print(res)


if __name__ == "__main__":
    # question_answer_prompt_template_example()
    multiple_questions_example()
