import jieba
from typing import List
from pypinyin import pinyin, Style, lazy_pinyin
import os
import functools

class Tokenizer:
    """
    分词器（使用 jieba 替代 LTP）
    """

    def __init__(self,
                 granularity: str = "word",
                 device: str = "cpu",
                 segmented: bool = False,
                 bpe: bool = False,
                 ) -> None:
        self.segmented = segmented
        self.granularity = granularity
        if self.granularity == "word":
            self.tokenizer = self.split_word
        elif self.granularity == "char":
            self.tokenizer = functools.partial(self.split_char, bpe=bpe)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "{:s}\nMode:{:s}\n}".format(str(self.__class__.__name__), self.granularity)

    def __call__(self,
                 input_strings: List[str]
                 ) -> List:
        if not self.segmented:
            input_strings = ["".join(s.split(" ")) for s in input_strings]
        results = self.tokenizer(input_strings)
        return results

    def split_char(self, input_strings: List[str], bpe=False) -> List:
        if bpe:
            from . import tokenization
            project_dir = os.path.dirname(os.path.dirname(__file__))
            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(project_dir,"data","chinese_vocab.txt"), do_lower_case=False)
        results = []
        for input_string in input_strings:
            if not self.segmented:
                segment_string = " ".join([char for char in input_string] if not bpe else tokenizer.tokenize(input_string))
            else:
                segment_string = input_string
            segment_string = segment_string.replace("[ 缺 失 成 分 ]", "[缺失成分]").split(" ")
            results.append([(char, "unk", pinyin(char, style=Style.NORMAL, heteronym=True)[0]) for char in segment_string])
        return results

    def split_word(self, input_strings: List[str]) -> List:
        """
        使用 jieba 分词（替代 LTP）
        """
        result = []
        for input_string in input_strings:
            if self.segmented:
                words = input_string.split(" ")
            else:
                words = list(jieba.cut(input_string))
            pos_tags = ["unk"] * len(words)
            pinyins = [lazy_pinyin(word) for word in words]
            result.append(list(zip(words, pos_tags, pinyins)))
        return result

if __name__ == "__main__":
    tokenizer = Tokenizer("word")
    print(tokenizer(["LAC是个优秀的分词工具", "百度是一家高科技公司"]))
