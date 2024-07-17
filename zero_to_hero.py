from langchain.llms import OpenAI


def main():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

    text = (
        "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers "
        "outdoor activities."
    )
    print(llm(text))


if __name__ == "__main__":
    main()
