# POS Tangging Bahasa Indonesia

Menggunakan CRFTagger dengan pretrained model berdasarkan data Fam Rashel (200rb-an token)

## Kode Pretrained Model

```python
from nltk.tag import CRFTagger

jumSample = 500000
namaFile = "./Indonesian_Manually_Tagged_Corpus.tsv"
with open(namaFile, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

pasangan = []
allPasangan = []

for line in lines[: min(jumSample, len(lines))]:
    if line == '':
        allPasangan.append(pasangan)
        pasangan = []
    else:
        kata, tag = line.split('\t')
        p = (kata,tag)
        pasangan.append(p)

ct = CRFTagger()
ct.train(allPasangan,'all_indo_man_tag_corpus_model_test.crf.tagger')
```

## Dataset POS-Tag Bahasa Indonesia

- https://github.com/UniversalDependencies/UD_Indonesian
- https://github.com/famrashel/idn-tagged-corpus
- http://www.panl10n.net/english/OutputsIndonesia2.htm
- https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989
