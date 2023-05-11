import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import CRFTagger
nltk.download('punkt')

# Contoh kalimat yang akan di lakukan POS Tagging
kalimat = "terus ya saya kan suka beli cimol jadi ini lagi main hp sambil beli cimol terus sih mamang bertanya mau dikasi bumbu apa dengan gesture nnyodorin bungkusan cimol lah saya tidak konsen karena lagi balas chat bukannya jawab saya malah salim cium jidat sama mamang cimol astaga"

# Tokenisasi kata dalam kalimat
words = word_tokenize(kalimat)

# Menggunakan CRFTagger dengan pretrained model berdasarkan data Fam Rasher (+-200rb token) sesuai dengan path yang telah ditentukan
ct = CRFTagger()
ct.set_model_file("./all_indo_man_tag_corpus_model.crf.tagger")
result = ct.tag(words)

# Menampilkan hasil POS Tagging
print(pd.DataFrame(result))