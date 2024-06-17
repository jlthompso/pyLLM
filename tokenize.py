import csv
import os.path
import re
import urllib.request

tmp_data_folder = 'tmp'
end_of_text_token = '<|endoftext|>'
unknown_text_token = '<|unk|>'


def main():
    if not os.path.exists(tmp_data_folder): os.mkdir(tmp_data_folder)

    data_source_paths = []
    with open('training_data_sources.csv', newline='') as f:
        for data_source in csv.DictReader(f):
            local_file_path = os.path.join(tmp_data_folder, data_source['filename'])
            if not os.path.isfile(local_file_path):
                urllib.request.urlretrieve(data_source['url'], local_file_path)
            data_source_paths.append(local_file_path)

    for data_source_path in data_source_paths:
        with open(data_source_path, 'r', encoding='utf-8') as f:
            preprocessed_text = re.split(r'([,.?_!"()\']|--|\s)', f.read())
            preprocessed_text = [token.strip() for token in preprocessed_text if token.strip()]
            tokens = sorted(set(preprocessed_text)) + [end_of_text_token, unknown_text_token]
            vocab = {token: id for id, token in enumerate(tokens)}
            tokenizer = SimpleTokenizer(vocab)

            text1 = "Hello, do you like tea?"
            text2 = "In the sunlit terraces of the palace."
            text = f" {end_of_text_token} ".join((text1, text2))
            print(text)
            print(tokenizer.encode(text))
            print(tokenizer.decode(tokenizer.encode(text)))


class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab_indexed_by_token = vocab
        self.vocab_indexed_by_id = {id: token for token, id in vocab.items()}

    def encode(self, text):
        preprocessed_text = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed_text = [token.strip() for token in preprocessed_text if token.strip()]
        preprocessed_text = [token if token in self.vocab_indexed_by_token else unknown_text_token for token in preprocessed_text]
        return [self.vocab_indexed_by_token[s] for s in preprocessed_text]

    def decode(self, token_ids):
        text = ' '.join([self.vocab_indexed_by_id[id] for id in token_ids])
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


if __name__ == '__main__':
    main()
