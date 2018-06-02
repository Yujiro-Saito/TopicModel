import glob
import re
import io,sys
import urllib.request
import gensim
from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#コーパス
texts = {}

#ファイル名一覧
paths = glob.glob('./KNBC_v1.0_090925/corpus2/*.tsv')

for path in paths:
    with open(path, "r", encoding="euc-jp") as f:
        #全ての記事群を読み込み
        tsv = f.read()

        #改行で読み込み IDと行
        for i, line in enumerate(tsv.split("\n")):
            if line == "":
                break
            #コラムタブで分ける
            columns = line.split("\t")
            index = columns[0].split("-")[0]
            if not index in texts:
                texts[index] = ""
                continue
            texts[index] = texts[index] + columns[1]


#形態素解析
char_filters = [UnicodeNormalizeCharFilter(),RegexReplaceCharFilter('\d+', '0')] 
tokenizer = Tokenizer(mmap=True)
token_filters = [POSKeepFilter(["名詞","形容詞","副詞","動詞"]), LowerCaseFilter(),ExtractAttributeFilter("base_form")]
analyzer = Analyzer(char_filters, tokenizer, token_filters)


#単語抽出とストップワード

stopwords = []
url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"

with urllib.request.urlopen(url) as response:
    stopwords = [w for w in response.read().decode().split('\r\n') if w != ""]


texts_words = {}

for k, v in texts.items():
    texts_words[k] = [w for w in analyzer.analyze(v)]


#辞書
dictionary = gensim.corpora.Dictionary(texts_words.values())
dictionary.filter_extremes(no_below=3, no_above=0.4)

#コーパス
corpus = [dictionary.doc2bow(words) for words in texts_words.values()]


#LDA Model 教師なし
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=4,id2word=dictionary,random_state=1)
print('topics: {}'.format(lda.show_topics()))
