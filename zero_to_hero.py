from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


def direct_llm_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

    text = (
        "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers "
        "outdoor activities."
    )
    print(llm(text))


def prompt_template_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    print(chain.run("eco-friendly water bottles"))


def conversation_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    conversation.predict(input="Tell me about yourself.")

    conversation.predict(input="What can you do?")
    conversation.predict(input="How can you help me with data analysis?")

    print(conversation)


if __name__ == "__main__":
    # direct_llm_example()
    # prompt_template_example()
    conversation_example()
