"""
TensorFlow Text: tokenizers, normalizers, BertTokenizer concepts.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Text Processing")
    print("=" * 50)

    try:
        import tensorflow_text as text
        print("tensorflow_text imported successfully")
    except ImportError:
        print("tensorflow_text not installed. Install: pip install tensorflow-text")
        return

    print("\nWhitespace tokenizer:")
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(["Hello world", "TensorFlow text"])
    print(f"  Input: ['Hello world', 'TensorFlow text']")
    print(f"  Tokens: {tokens}")

    print("\nUnicode script tokenizer:")
    script_tokenizer = text.UnicodeScriptTokenizer()
    script_tokens = script_tokenizer.tokenize(["Hello, world!"])
    print(f"  Input: ['Hello, world!']")
    print(f"  Tokens: {script_tokens}")

    print("\nCase folding normalizer:")
    normalizer = text.case_fold_utf8(["Hello WORLD"])
    print(f"  Input: ['Hello WORLD']")
    print(f"  Normalized: {normalizer}")

    print("\nBertTokenizer concepts (vocab-based):")
    vocab = ["[UNK]", "[CLS]", "[SEP]", "hello", "world", "tf", "text"]
    vocab_file = "/tmp/tf_text_vocab.txt"
    with open(vocab_file, "w") as f:
        f.write("\n".join(vocab))
    bert_tokenizer = text.BertTokenizer(vocab_file, lower_case=True)
    bert_tokens = bert_tokenizer.tokenize(["hello world"])
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Tokens for 'hello world': {bert_tokens}")

    print("\nTF Text demo complete.")

if __name__ == "__main__":
    main()
