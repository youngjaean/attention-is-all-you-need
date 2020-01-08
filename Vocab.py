import sys, os, argparse, datetime, time, re, collections
import pandas as pd
import csv
import sentencepiece as spm

def build_corpus(infile, outfile):
    csv.field_size_limit(sys.maxsize)#포인트 사이즈 만큼 최대 필드 크기를 반환
    SEPARATOR = u"\u241D"
    df = pd.read_csv(infile, sep=SEPARATOR, engine='python')

    with open(outfile, 'w') as f:
        for index, row in df.iterrows(): #첫번째 변수는 index를 받고 두번째는 행에 하나씩 접근
            f.write(row["text"].lower())
            f.write("\n\n\n\n")
            print(f"build corpus ... {index + 1} / {len(df)}", end="\r")
        print()
        return outfile



def build_vocab(args):
    spm.SentencePieceTrainer.train(
        f"--input={args.corpus} --model_prefix={args.prefix} --vocab_size={args.vocab_size + 7}" + 
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +
        " --unk_id=1 --unk_piece=[UNK]" +
        " --bos_id=2 --bos_piece=[BOS]" +
        " --eos_id=3 --eos_piece=[EOS]" +
        " --user_defined_symbols=[SEP],[CLS],[MASK]")
# input: 입력 corpus
# prefix: 저장할 모델 이름
# vocab_size: vocab 개수 
# max_sentence_length: 문장의 최대 길이
# pad_id, pad_piece: pad token id, 값
# unk_id, unk_piece: unknown token id, 값
# bos_id, bos_piece: begin of sentence token id, 값
# eos_id, eos_piece: end of sequence token id, 값
# user_defined_symbols: 사용자 정의 토큰

def load_vocab(file):
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default='kowiki', type=str, required=False)
    parser.add_argument('--vocab_size', default=8000, type=int, required=False)
    args = parser.parse_args()

    args.corpus = 'kowiki/kowiki.txt'
    if not os.path.isfile(args.corpus): #파일 존재 확인
        build_corpus("data/kowiki.csv", args.corpus)
    build_vocab(args)
