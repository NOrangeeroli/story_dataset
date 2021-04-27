import json
from transformers import T5Tokenizer
# dir_ = "./"
# tokenizer = T5Tokenizer("%s/vocab.txt"%dir_)
# tokenizer.save_vocabulary(dir_)

import urllib.request
import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
# model = io.BytesIO()
# with urllib.request.urlopen(
#     'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
# ) as response:
#   spm.SentencePieceTrainer.train(
#       sentence_iterator=response, model_writer=model, vocab_size=1000)
# print(type(model))
# with open('out.model', 'wb') as f:
#   f.write(model.getvalue())


# Directly load the model from serialized model.
# sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
sp = spm.SentencePieceProcessor()
sp.Load("./spiece.model")
print(sp.get_piece_size())
vocab = {}
for i in range(sp.get_piece_size()):
    vocab[sp.id_to_piece(i)] = i
with open("vocab.json", "w") as fout:
    json.dump(vocab, fout, ensure_ascii=False, indent=4)
# print(sp.encode('两只老虎，两只老虎，跑得快，跑得快', out_type=str))
# tokenizer = T5Tokenizer("./spiece.model")
# tokenizer.save_vocabulary("./new")
exit()


with open("spiece_example.model", "rb") as fin:
    for line in fin:
        print(line)


with open("vocab.txt") as fin:
    data = [line.strip() for line in fin]
    data_dict = {}
    for i, d in enumerate(data):
        data_dict[d] = i
with open("vocab.json", "w") as fout:
    json.dump(data_dict, fout, indent=4, ensure_ascii=False)

