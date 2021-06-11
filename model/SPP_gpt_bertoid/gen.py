from logging import log
import json
import traceback
import sys
from transformers import GPT2Model,BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import numpy as np
from run_clm import MyModel
from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        # BertForSequenceClassification,
        AutoTokenizer,    
        # T5Tokenizer,
        # AutoTokenizer,
        # AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )
import random

# print(torch.cuda.device_count())
device = "cuda:6"
print("using %s"%device)
model_name_path = "./output/checkpoint-1000/"
print(model_name_path)
ckpt_list = False
# ckpt_list = True
PPL, generation = False, True
# PPL, generation = True, False
name = "data/data_gpt"
with open("./%s/test.txt"%name, "r") as fin:
    data = [x.strip() for x in fin.readlines()]
    ipt = [d.split('[S]')[0] for d in data]
    opt = [d.split('[S]')[1] for d in data]

# with open("./%s/test.source"%(name), "r") as fin:
#     ipt = [line.strip() for line in fin]
# with open("./%s/test.target"%(name), "r") as fin:
#     opt = [line.strip() for line in fin]

import sys
from unicodedata import category
chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))
def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def pro(token_list, tokenizer):
    # string = tokenizer.convert_ids_to_tokens(token_list, skip_special_tokens=False)
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<pad>", "").strip()
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    string = " ".join(string.strip().split())
    if "zh" in name:
        string = strB2Q(string)
    return string


if generation:
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)

    # for tok in [    1,    19, 32099,    19, 32098]:
    #     print(tokenizer.decode([tok]))
    # exit()

    pad_token_id = tokenizer.pad_token_id

    tokenizer.add_special_tokens({"additional_special_tokens": ["[P]"]})
    sp_id = tokenizer.convert_tokens_to_ids("[P]")
    # print(tokenizer.encode("<extra_id_0><extra_id_1>"))

    # model = BartModel.from_pretrained('./bart-base', return_dict=True)
    config = AutoConfig.from_pretrained(model_name_path)

    model = MyModel(config=config, model_name_or_path = model_name_path, tokenizer=tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    print(tokenizer.encode("我[P]和[MASK]你"))
    for t in [101, 2769, 21128, 1469, 103, 872, 102]:
        print(tokenizer.decode([t]))
    # print(tokenizer.decode([32099]))
    # print(tokenizer.decode([2]))
    # exit()

    # available_list = []
    # for i in range(1, 100):
    #     # print(tokenizer.tokenize("<extra_id_%d>"%i))
    #     available_list += tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<extra_id_%d>"%i))
    # # exit()

    # def get_label(gen_list):
    #     begin_id = 0
    #     for id_, k in enumerate(gen_list):
    #         if k == available_list[0]:
    #             begin_id = id_

    #     for k in range(1, 512):
    #         if gen_list[begin_id+k] != available_list[k]:
    #             label = available_list[k-1]
    #             break
    #     return label


    file_out = "./result/%s.txt"%(model_name_path.replace("/", "_").replace(".", ""))
    print("write to %s"%file_out)
    num = 0
    
    with open(file_out, "w") as fout:
        batch_size = 1
        st, ed = 0, 0
        all_loss = []
        with torch.no_grad():
            for ip in ipt:
                input_ids = tokenizer(ipt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
                opt = model(input_ids).logits
                import pdb;pdb.set_trace()
            '''
            while ed < len(ipt):
                st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
                
                input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
                # gen = model.generate(input_ids, do_sample=False, max_length=512, decoder_start_token_id=1) # 0 for mt5 and 1 for cst5# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                # gen = model.generate(input_ids, do_sample=False, num_beams=1, max_length=512, decoder_start_token_id=19, early_stopping=True)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                # gen = sample_sequence(input_ids=input_ids, model=model, max_length=128, temperature=0.7, top_k=40)

                outputs = motokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)del(input_ids, output_hidden_states=True, return_dict=True)

                # [batch_size, length, hidden_size]
                encoder_hidden_states = outputs.hidden_states[-1] #outputs["encoder_last_hidden_state"]
                # [batch_size, length]
                mask1 = torch.eq(input_ids, torch.tensor(sp_id).to(input_ids.device)).float()
                mask2 = torch.eq(input_ids, torch.tensor(tokenizer.mask_token_id).to(input_ids.device)).float()
                # [batch_size, length]
                logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
                logits -= (1 - mask2) * (1e20)

                # [batch_size]
                # label = tgt_ids[:, 0]
                # [batch_size, length]
                # mask3 = torch.eq(torch.cumsum(torch.eq(torch.cumsum(mask2, 1), labels[:, None]).float(), 1), 1).float()
                

                # # [batch_size, length, hidden_size]
                # encoder_hidden_states = outputs["encoder_last_hidden_state"]
                # # [batch_size, length]
                # mask1 = torch.eq(src_ids, torch.tensor(tokenizer.bos_token_id).to(src_ids.device)).float()
                # mask2 = 1 - torch.lt(src_ids, 32000).float()
                # # [batch_size, length]
                # logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
                # logits -= (1 - mask2) * (1e20)
                # # lprobs = torch.nn.functional.softmax(logits, dim=-1)

                # [batch_size]
                # label = tgt_ids[:, 0]
                # [batch_size, length]
                # mask3 = torch.eq(src_ids, label[:, None]).float()
                # loss_cls = -torch.mean(torch.log(torch.sum(lprobs*mask3, 1)+1e-20))

                for idx_, (ip, op, mk, truth) in enumerate(zip(input_ids, logits, mask2, opt[st:ed])):
                    if True:
                        pred = op.to("cpu").numpy().tolist()
                        id_ = torch.cumsum(mk, 0).to("cpu").numpy().tolist()
                        pred_id = int(id_[np.argmax(pred)])

                        # pred_id = int(ip[id_].to("cpu").numpy())
                        # print(pred, tokenizer.convert_ids_to_tokens(pred))
                        # pred_id = get_label(pred)

                        # label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(truth))
                        import pdb;pdb.set_trace()
                        label_id = int(truth)#label[0]
                        # print(pred_id, label_id)
                        
                        # print(label, tokenizer.tokenize(truth))
                        # # assert len(label) == 1
                        # # label = label[0]
                        # label_id = get_label(label)

                        if pred_id == label_id:
                            num += 1
                            # print("True")
                            # print(tokenizer(ipt[st+idx_], return_tensors="pt").input_ids.size())
                        # print("="*10)
                    '''     
                    # for k in pred:
                    #     if k in [250099, 250098]:
                    #         if ((k == 250098) and ("1" in truth)) or ((k == 250099) and ("0" in truth)):
                    #             num += 1
                    #         break

                    # for k in pred:
                    #     if k in available_list:
                    #         # if ((k == 32098) and ("1" in truth)) or ((k == 32099) and ("0" in truth)):
                    #         if k == label:
                    #             num += 1
                    #         break
                    # print(truth)
                    # print(tokenizer.decode(op))
                    # fout.write(pro(op, tokenizer)+"\n")
                    # fout.write(pro(ip, tokenizer) + "|||" + pro(op, tokenizer)+"\n")
                    # print(pro(ip, tokenizer))
                    # print(pro(op, tokenizer))
                    # print(truth)
                    # print("="*10)
                    # print(tokenizer.tokenize(truth))
                    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(truth))))
                    # print("="*5)
        print(num / len(opt))
