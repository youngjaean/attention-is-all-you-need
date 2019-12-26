import torch.nn as nn
from torch.autograd import Variable
import torch
from LabelSmooth import LabelSmoothing
from make_modle import make_model
from make_modle import Batch
from torchtext import data, datasets
from run_config    import greedy_decode
from Optima import NoamOpt
from config import batch_size_fn, run_epoch
from torchtext import data, datasets

#     TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))



device = torch.device("cuda:0")
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model)
None

# if False:
#     model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#     for epoch in range(10):
#         model_par.train()
#         run_epoch((rebatch(pad_idx, b) for b in train_iter), 
#                   model_par, 
#                   MultiGPULossCompute(model.generator, criterion, 
#                                       devices= device, opt=model_opt))
#         model_par.eval()
#         loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
#                           model_par, 
#                           MultiGPULossCompute(model.generator, criterion, 
#                           devices= device, opt=None))
#         print(loss)
# else:
#     model = torch.load("iwslt.pt")

# for i, batch in enumerate(valid_iter):
#     src = batch.src.transpose(0, 1)[:1]
#     src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
#     out = greedy_decode(model, src, src_mask, 
#                         max_len=60, start_symbol=TGT.vocab.stoi[""])
#     print("Translation:", end="\t")
#     for i in range(1, out.size(1)):
#         sym = TGT.vocab.itos[out[0, i]]
#         if sym == "</s>": break
#         print(sym, end =" ")
#     print()
#     print("Target:", end="\t")
#     for i in range(1, batch.trg.size(0)):
#         sym = TGT.vocab.itos[batch.trg.data[i, 0]]
#         if sym == "</s>": break
#         print(sym, end =" ")
#     print()
#     break

# torch.cuda.empty_cache() 

# model, SRC, TGT = torch.load("en-de-model.pt")
# model.eval()
# sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
# src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
# src = Variable(src)
# src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
# out = greedy_decode(model, src, src_mask, 
#                     max_len=60, start_symbol=TGT.stoi["<s>"])
# print("Translation:", end="\t")
# trans = "<s> "
# for i in range(1, out.size(1)):
#     sym = TGT.itos[out[0, i]]
#     if sym == "</s>": break
#     trans += sym + " "
# print(trans)

import fire
if __name__ == '__main__':
    fire.Fire()