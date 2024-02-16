from transformers import PreTrainedTokenizer
from konlpy.tag import Kkma, Okt, Komoran, Hannanum, Mecab

class CustomedTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_name):
        self.tokenizer = self._select_tokenizer(tokenizer_name)
        super().__init__()

    def _select_tokenizer(self, tokenizer_name):
        if tokenizer_name == 'okt':
            return Okt()
        elif tokenizer_name == 'mecab':
            return Mecab()
        elif tokenizer_name == "komoran":
            return Komoran()
        elif tokenizer_name == "hannanum":
            return Hannanum()
        elif tokenizer_name == "kkma":
            return Kkma()
        else:
            raise ValueError("Unsupported tokenizer type")

    def tokenize(self, text):
        return self.tokenizer.morphs(text)