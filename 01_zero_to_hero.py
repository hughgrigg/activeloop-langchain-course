"""
Requires the following env vars:
 - OPENAI_API_KEY
 - ACTIVELOOP_TOKEN
 - ACTIVELOOP_ORG_ID
 - GOOGLE_API_KEY
 - GOOGLE_CSE_ID
"""

import os

from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import DeepLake


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


def deeplake_dataset_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    texts = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638",
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)

    my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    db.add_documents(docs)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )

    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.run,
            description="Useful for answering questions.",
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response = agent.run("When was Napoleon born?")
    print(response)


def deeplake_existing_dataset_example():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    texts = [
        "Lady Gaga was born in 28 March 1986",
        "Michael Jeffrey Jordan was born in 17 February 1963",
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)

    db.add_documents(docs)

    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )

    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.run,
            description="Useful for answering questions.",
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response = agent.run("When was Michael Jordan born?")

    print(response)


def agent_tools_google_search_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name="google-search",
            func=search.run,
            description="useful for when you need to search google to answer questions about current events",
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=6,
    )

    response = agent("What's the latest news about the Mars rover?")

    print(response["output"])


def agent_multi_tools_example():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    prompt = PromptTemplate(
        input_variables=["query"],
        template="Write a summary of the following text: {query}",
    )

    summarize_chain = LLMChain(llm=llm, prompt=prompt)

    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for finding information about recent events",
        ),
        Tool(
            name="Summarizer",
            func=summarize_chain.run,
            description="useful for summarizing texts",
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response = agent(
        "What's the latest news about the Mars rover? Then please summarize the results."
    )
    print(response["output"])


if __name__ == "__main__":
    # direct_llm_example()
    # prompt_template_example()
    # conversation_example()
    # deeplake_new_dataset_example()
    # deeplake_existing_dataset_example()
    # agent_tools_google_search_example()
    agent_multi_tools_example()
