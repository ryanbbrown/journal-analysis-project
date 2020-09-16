### TEXT GENERATION

############################################################################
# imports + setup
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import pandas as pd
from fastai.text import *

# reading
path = '/notebooks/my-data'
df = pd.read_csv('/notebooks/my-data/corpus.csv')
df = df[['date','entry']]


############################################################################
# language model
bs=16
df = df[['entry']]
data_lm = (TextList.from_df(df=df, path=path)
           .split_by_rand_pct(0.1, seed=50)
           .label_for_lm()
           .databunch(bs=bs))
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
learn.save_encoder('fine_tuned_encoder')


############################################################################
# text generation
TEXT = "Okay"
N_WORDS = 200
N_SENTENCES = 3

for i in range(0,N_SENTENCES):
    print('\n')
    print(learn.predict(TEXT, N_WORDS, temperature=0.75))
    
TEXT = "My dorm"
N_WORDS = 200
N_SENTENCES = 3

for i in range(0,N_SENTENCES):
    print('\n')
    print(learn.predict(TEXT, N_WORDS, temperature=0.75))


    # nucleus predict
'''Paper that mentions it: https://arxiv.org/abs/1904.09751'''
'''Source: https://forums.fast.ai/t/interesting-read-on-neural-text-de-generation/45211/2'''
def nucleus_predict(self, text:str, n_words:int=1, top_p:float=0.9, sep:str=' ', decoder=decode_spec_tokens):
    ds = self.data.single_dl.dataset
    self.model.reset()
    xb, yb = self.data.one_item(text)
    new_idx = []
    for _ in range(n_words):
        res  = self.pred_batch(batch=(xb,yb))[0][-1]
        nucleus = []
        sum = 0
        for p in res:
            sum += p
            nucleus.append(p)
            if sum > top_p:
                break
        idx = torch.multinomial(torch.FloatTensor(nucleus), 1).item()
        new_idx.append(idx)
        xb = xb.new_tensor([idx])[None]
    return text + sep + sep.join(decoder(self.data.vocab.textify(new_idx, sep=None)))

    # beam search
learn.beam_search(TEXT, N_WORDS, temperature=0.75)
