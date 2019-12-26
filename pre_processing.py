#-*- coding: utf-8 -*-

import pandas as pd

in_file = "web-crawler/kowiki/kowiki_20191223.csv"
out_file = "kowiki/kowiki.txt"
SEPARATOR = u"\u241D"
SEPARATOR = SEPARATOR.encode('utf-8') #인코딩 문제 해결
df = pd.read_csv(in_file, sep=SEPARATOR.decode('utf-8'), engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자

#본문 저장 및 각각의 문서 구분 


import sentencepiece as spm

corpus = "kowiki/kowiki.txt"
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
