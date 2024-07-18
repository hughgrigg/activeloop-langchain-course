from transformers import AutoTokenizer


def auto_tokenizer_example():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

    print("Tokens:   ", tokenizer.convert_ids_to_tokens(token_ids))
    print("Token IDs:", token_ids)


if __name__ == "__main__":
    auto_tokenizer_example()
