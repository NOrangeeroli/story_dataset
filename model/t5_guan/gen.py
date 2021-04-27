from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import numpy as np
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )
import random
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.topk(logits, k=k)
        min_values = torch.min(values, -1)[:, None]# values[:, -1, tf.newaxis]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       torch.eq(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, axis=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits), axis=-1)
    cumulative_position = torch.sum((cumulative_probs <= p).to(torch.int32), axis=-1) - 1
    indices = torch.stack([
        torch.arange(0, batch).to(device),
        # number of indices to include
        torch.max(cumulative_position, torch.zeros([batch], dtype=cumulative_position.dtype).to(device)),
    ], axis=-1)
    min_values = gather_nd(sorted_logits, indices).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(input_ids, model, max_length, temperature=0.7, top_p=0.9, no_sample=False):
    batch_size = input_ids.size()[0]
    decoder_input_ids = torch.tensor([1 for _ in range(batch_size)])[:, None].to(device)
    # tokens_embed = model.transformer.get_input_embeddings()
    for i in range(max_length):
        logits = model(input_ids, decoder_input_ids=decoder_input_ids)["logits"]
        logits = logits[:, -1, :] / temperature

        if no_sample:
            prev = torch.topk(logits, 1)[1]
        else:
            logits = top_p_logits(logits, p=top_p)
            probs = torch.nn.functional.softmax(logits)
            prev = torch.multinomial(probs, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, prev], 1)
    return decoder_input_ids

# print(torch.cuda.device_count())
device = "cuda:1"
print("using %s"%device)

model_name_path = "./model/t5-small_train21/best_tfmr"
print(model_name_path)
ckpt_list = False
# ckpt_list = True
PPL, generation = False, True
# PPL, generation = True, False
name = "data/data_train"
with open("./%s/test.source"%name, "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/test.target"%name, "r") as fin:
    opt = [line.strip() for line in fin]

def pro(token_list, tokenizer):
    # string = tokenizer.convert_ids_to_tokens(token_list, skip_special_tokens=False)
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].strip()
    return string

tag_list = ["外国文学", "生活科普", "武侠仙侠", "都市言情", "玄幻仙侠"]
if generation:
    if not ckpt_list:
        model_name_path_dict = {"best": model_name_path}
    else:
        model_name_path_dict = {}
        for _, dd, _ in os.walk(model_name_path):
            for d in dd:
                if d.startswith("val_avg_loss="):
                    model_name_path_dict[d.split("=")[-1].split(".")[0]] = "%s/%s"%(model_name_path, d)
            break
    for name in sorted(model_name_path_dict.keys()):
        try:
            if ckpt_list and (int(name) < 40 or int(name) > 70):
                continue
            # if int(name) % 10 != 0:
            #     continue
        except:
            continue
        print("loading model %s"%name)

        tmp_model_name_path = model_name_path_dict[name]
        tokenizer = AutoTokenizer.from_pretrained(tmp_model_name_path)
        pad_token_id = tokenizer.pad_token_id

        # model = BartModel.from_pretrained('./bart-base', return_dict=True)
        model = T5ForConditionalGeneration.from_pretrained(tmp_model_name_path, return_dict=True).to(device)
        model.eval()

        file_out_dir = "./result/%s"%model_name_path.replace(".", "").replace("/", "")
        if not os.path.exists(file_out_dir):
            os.mkdir(file_out_dir)
        file_out = "%s/%s.txt"%(file_out_dir, name)
        print("write to %s"%file_out)
        with open(file_out, "w") as fout:
            batch_size = 32
            st, ed = 0, 0
            all_loss = []
            with torch.no_grad():
                while ed < len(ipt):
                    st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
                    
                    # for i in range(st, ed):
                    #     if "<eod>" in ipt[i]:
                    #         ipt[i] = "玄幻仙侠<eod>"+ipt[i].strip().split("<eod>")[1]
                    #     else:
                    #         ipt[i] = "玄幻仙侠<eod>"+ipt[i].strip()


                    input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
                    gen = model.generate(input_ids, do_sample=False, decoder_start_token_id=1, max_length=512, early_stopping=True)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                    # gen = model.generate(input_ids, do_sample=False, num_beams=1, max_length=512, early_stopping=True)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                    # gen = sample_sequence(input_ids=input_ids, model=model, max_length=512, temperature=0.7, top_p=0.9, no_sample=True)
                    for ip, op, truth in zip(input_ids, gen, opt[st:ed]):
                        # fout.write(pro(ip, tokenizer) + "|||" + pro(op, tokenizer)+"\n")
                        print(pro(ip, tokenizer) + "|||" + pro(op, tokenizer))
                        # print(tokenizer.tokenize(truth))
                        # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(truth))))
                        # print("="*5)

if PPL:
    batch_size = 10

    if not ckpt_list:
        model_name_path_dict = {"best": model_name_path}
    else:
        model_name_path_dict = {}
        for _, dd, _ in os.walk(model_name_path):
            for d in dd:
                if d.startswith("val_avg_loss="):
                    model_name_path_dict[d.split("=")[-1].split(".")[0]] = "%s/%s"%(model_name_path, d)
            break
    for name in sorted(model_name_path_dict.keys()):
        try:
            if ckpt_list and int(name) < 35:
                continue
        except:
            continue
        print("loading model %s"%name)
        tmp_model_name_path = model_name_path_dict[name]
        tokenizer = BartTokenizer.from_pretrained(tmp_model_name_path)
        pad_token_id = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id

        # model = BartModel.from_pretrained('./bart-base', return_dict=True)
        model = BartForConditionalGeneration.from_pretrained(tmp_model_name_path, return_dict=True).to(device)
        model.eval()
        st, ed = 0, 0
        all_loss = []
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
            with torch.no_grad():
                src_ids = input_ids
                tgt_ids = tokenizer(opt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
                # tgt_ids = torch.cat([torch.zeros([batch_size, 1], dtype=tgt_ids.dtype), tgt_ids], 1)
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
                # print(src_ids, tokenizer.decode(src_ids[0], skip_special_tokens=False))
                # print(tgt_ids, tokenizer.decode(tgt_ids[0], skip_special_tokens=False))
                outputs = model(src_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
                lm_logits = outputs["logits"]
                # print(src_ids.size(), lm_logits.size(), decoder_input_ids.size())


                tmp_batch_size = lm_logits.size()[0]
                pad_pos = torch.eq(tgt_ids, pad_token_id).to(torch.float)
                sen_pos = torch.eq(tgt_ids, mask_token_id).to(torch.float)
                dis_pos = torch.cat([torch.zeros([tmp_batch_size, 1]).to(sen_pos.device), sen_pos[:, :-1]], 1)
                loss_mask = 1 - (pad_pos + sen_pos + dis_pos)
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

                # assert lm_logits.shape[-1] == self.vocab_size
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
                loss = torch.sum(loss * loss_mask.view(-1)) / (torch.sum(loss_mask) + 1e-20)
                all_loss.append(loss.cpu().numpy())


                # # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                # ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

                # # assert lm_logits.shape[-1] == self.vocab_size
                # loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
                # all_loss.append(loss.cpu().numpy())
        print(name, "perplexity:", np.exp(np.mean(all_loss)))

    # all_loss_dict = {}
    # for i, l in enumerate(all_loss):
    #     all_loss_dict[i] = l
    # idx_list = sorted(all_loss_dict, key=all_loss_dict.get, reverse=True)
    # with open("./ppl.txt", "w") as fout:
    #     for idx in idx_list:
    #         fout.write("%d|||%.4f|||%s|||%s\n"%(idx, all_loss_dict[idx], ipt[idx], opt[idx]))