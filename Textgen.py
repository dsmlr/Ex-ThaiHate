import os
import time
import argparse
# from turtle import pd
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
# from model import BartModel
# from model import BartForMaskedLM
# from transformers import BartTokenizer
# from transformers.modeling_bart import make_padding_mask
# from transformers import BartTokenizer
# from transformers.models.bart.modeling_bart import make_padding_mask
from utils.helper import make_padding_mask
# from classifier.textcnn import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator
import pickle
import time
import random

# from transformers.modeling_utils import PreTrainedModel, unwrap_model
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast,MBartConfig
# from transformers import (
#     MBart50TokenizerFast,
#     AdamW
# )
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# from transformers.models.mbart.configuration_mbart import MBartConfig

# from transformers.models.mbart.modeling_mbart import (
#     MBartPreTrainedModel,
#     MBartDecoder,
#     MBartLearnedPositionalEmbedding,
#     MBartEncoderLayer,
#     shift_tokens_right,
#     _expand_mask 
# )


from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
	Seq2SeqModelOutput
)
from transformers import MBartForConditionalGeneration

device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(0)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


    # base = BartModel.from_pretrained("facebook/bart-base")
    # model = BartForMaskedLM.from_pretrained('facebook/bart-base', config=base.config)
    # # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # # base = MultimodalBartModel.from_pretrained("facebook/bart-base")
    # # model = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base', config=base.config)
    # model.to(device).train()
    df=pd.read_csv('dataset.csv')
    device = 'cuda' if cuda.is_available() else 'cpu'
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    print("Model loaded...\n")
    model.to(device)

    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                     src_lang="th_TH",
                                                     tgt_lang="en_XX")
    #Creating train,test and validation set
    x_train,x_test,y_train,y_test=train_test_split(df.index.to_list(),df.Hatespeech_Level.to_list(),test_size=0.20)
    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.10)
    
    # with open("/home/sysadm/Nilabja_summer/New-Folder/to_hindi/SBIC.v2.trn.post.csv", "rb") as fp:
    #     traintext = pickle.load(fp)
    # with open("trainCMlabelAndSSAndSpan", "rb") as fp:
    #     trainlabels = pickle.load(fp)
    traintext = df.iloc[x_train]
    traintext = traintext["Message"].tolist()
    task='hatx-hate-emo-sa'
    print('training for '+task+' ...')
    #Creating train labels
    trainlabels = df.iloc[x_train]
    hate = trainlabels['hatx'].tolist()
    hs=trainlabels['Hatespeech_Level'].to_list()
    
    d={'-2':'HATE','-1':'HATE','0':'NONHATE','1':'NONHATE'}
    hs=[d[str(hs[i])] for i in range(len(hate))]
    emotion = trainlabels['Emotion'].tolist()
    sentiment = trainlabels['Sentiment'].tolist()
    trainlabels = [str(hate[i])+' '+str(hs[i])+' '+str(sentiment[i])+' '+str(emotion[i]) for i in range(len(hate))]
    # print(len(traintext))
    # print(traintext[0])
    # print(trainlabels[0])
    # time.sleep(120)
    for i in range(len(traintext)):
        traintext[i]=str(traintext[i])
    for i in range(len(trainlabels)):
        trainlabels[i]=str(trainlabels[i])
    # time.sleep(120)

    # with open("", "rb") as fp:
    #     validtext = pickle.load(fp)
    # with open("validCMlabelAndSSAndSpan", "rb") as fp:
    #     validlabels = pickle.load(fp)
    #Creating validatin labels
    validtext = df.iloc[x_val]
    validtext = validtext["Message"].tolist()
    validlabels =df.iloc[x_val]
    # 	validlabels = validlabels[task].tolist()[6078:6837]
    hs=validlabels['Hatespeech_Level'].to_list()
    hate = validlabels['hatx'].tolist()
    d={'-2':'HATE','-1':'HATE','0':'NONHATE','1':'NONHATE'}
    hs=[d[str(hs[i])] for i in range(len(hate))]
    
    emotion = validlabels['Emotion'].tolist()
    sentiment = validlabels['Sentiment'].tolist()
    validlabels = [str(hate[i])+' '+str(hs[i])+' '+str(sentiment[i])+' '+str(emotion[i]) for i in range(len(hate))]


    for i in range(len(validtext)):
        validtext[i]=str(validtext[i])
    for i in range(len(validlabels)):
        validlabels[i]=str(validlabels[i])


    # print(len(validtext))



    trainsrc_seq, traintgt_seq = [], []
    max_len=30

    # for k in range(len(traintext)):
    f1 = traintext
    f2 = trainlabels
    index = [i for i in range(len(f1))]
    random.shuffle(index)
    index = index[:int(len(index) * 1.0)]
    for i, (s, t) in enumerate(zip(f1, f2)):
        if i in index:
            # print(s)
            if len(s)==0:
                continue
            s = tokenizer.encode(s)
            t = tokenizer.encode(t)
            s = s[:min(len(s) - 1, max_len)] + s[-1:]
            t = t[:min(len(t) - 1, max_len)] + t[-1:]
            trainsrc_seq.append(s)
            traintgt_seq.append([tokenizer.bos_token_id]+t)


    validsrc_seq, validtgt_seq = [], []
    max_len=30
    # for k in range(len(validtext)):
    f1 = validtext
    f2 = validlabels
    index = [i for i in range(len(f1))]
    random.shuffle(index)
    index = index[:int(len(index) * 1.0)]
    for i, (s, t) in enumerate(zip(f1, f2)):
        if i in index:
            s = tokenizer.encode(s)
            t = tokenizer.encode(t)
            s = s[:min(len(s) - 1, max_len)] + s[-1:]
            t = t[:min(len(t) - 1, max_len)] + t[-1:]
            # s[0] = domain
            validsrc_seq.append(s)
            validtgt_seq.append([tokenizer.bos_token_id]+t)

    train_loader, valid_loader = BARTIterator(trainsrc_seq, traintgt_seq,
                                              validsrc_seq, validtgt_seq)

    print('done')


    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), 1e-5, 10000)



    tab = 0
    eval_loss = 1e8
    total_loss_ce = []
    # total_loss_sc = []
    total_loss_co = []
    start = time.time()
    train_iter = iter(train_loader)
    #Training phase
    for step in range(1, 100001):
        print('current {}'.format(step))

        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src, tgt = map(lambda x: x.to(device), batch)
        src_mask = make_padding_mask(src, tokenizer.pad_token_id)
        src_mask = 1 - src_mask.long() if src_mask is not None else None
        logits = model(src, attention_mask=src_mask, decoder_input_ids=tgt)[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt[..., 1:].contiguous()
        loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
        total_loss_ce.append(loss_ce.item())

        loss_sc, loss_co = torch.tensor(0), torch.tensor(0)
        # if opt.sc and (200 < step or len(train_loader) < step):
        #     idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
        #     loss_sc = cal_sc_loss(logits, idx, cls, tokenizer, opt.style)
        #     total_loss_sc.append(loss_sc.item())
	
	#Reinforcement learning incorporated
        if (10000 < step or len(train_loader)< step):
            # print('RL')
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
            loss_co = cal_bl_loss(logits, tgt, idx, tokenizer)
            total_loss_co.append(loss_co.item())

        optimize(optimizer, loss_ce + loss_co)

        if step % 1000 == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            print('[Info] steps {:05d} | loss_ce {:.4f} | '
                  'loss_co {:.4f} | lr {:.6f} | second {:.2f}'.format(
                step, np.mean(total_loss_ce),
                np.mean(total_loss_co), lr, time.time() - start))
            total_loss_ce = []
            # total_loss_sc = []
            total_loss_co = []
            start = time.time()


# 		if  step%10000==0:
# 			torch.save(model.state_dict(), 'checkpoint-bart-Thai/{}.chkpt'.format(
# 			        task+str(step)))

# if ((len(train_loader) > 200
#      and step % 200 == 0)
#         or (len(train_loader) < 200
#             and step % len(train_loader) == 0)):
#     valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn,
#                                      tokenizer, step)
# if eval_loss >= valid_loss:
#     torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}.chkpt'.format(
#         opt.model, opt.dataset, opt.order, opt.style))
#         print('[Info] The checkpoint file has been updated.')
#         eval_loss = valid_loss
#         tab = 0
#     else:
#         tab += 1
#     if tab == opt.patience:
#         exit()
    #Creating labels for testing
    test_file = df.iloc[x_test]
    print('testing for '+task+' ...')
    post = test_file["Message"].tolist()
    # label = test_file[task].tolist()[6837:]
    hate = test_file['hatx'].tolist()
    hs=test_file['Hatespeech_Level'].to_list()
#     hate = test_file['hatx'].tolist()
    d={'-2':'HATE','-1':'HATE','0':'NONHATE','1':'NONHATE'}
    hs=[d[str(hs[i])] for i in range(len(hate))]
#     device='cpu'
#     model.to(device)
    model.eval()
    emotion = test_file['Emotion'].tolist()
    sentiment = test_file['Sentiment'].tolist()
    label = [str(hate[i])+' '+str(hs[i])+' '+str(sentiment[i])+' '+str(emotion[i]) for i in range(len(hate))]
    preds=[]
    c=0
    for text in post:
        c+=1
        try:
            src=tokenizer.encode(text,return_tensors='pt')
            generated_ids=model.generate(src.to(device),max_length=128)
            text=[tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=False) for g in generated_ids][0]
            print(text)
            # print(text)
            preds.append(text)
        except:
            preds.append('NO OUTPUT')
            print(c)
    df=pd.DataFrame({'original_'+task:label,'predicted_'+task:preds})
    df.to_csv('100000'+'ep_'+task+'_acc.csv',index=False)
    print(len(preds))


if __name__ == "__main__":
    main()
