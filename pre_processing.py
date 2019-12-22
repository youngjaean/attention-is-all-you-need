import pandas as pd

int_file = 'web-crawler/kowiki-latest-pages-meta-current.xml.bz2gxuxh176.tmp'
out_file = 'kowiki/kowiki.txt'
SEPARATOR = u"\u241D"
df = pd.read_csv(int_file, sep=SEPARATOR, engine="python")
with open(out_file, 'w') as f:
    for index, row in df.iterrows():
        f.write(row['text'])
        f.write("\n\n\n\n") 

#본문 저장 및 각각의 문서 구분 


import sentencepiece as spm

corpus = "kowiki.txt"
prefix = 'kowiki'

vocab_size = 8000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰