import torch
import torch.nn.functional as F
import os
import json
import re
import argparse
import numpy as np
from tqdm import trange
from transformers import GPT2LMHeadModel,BertTokenizer

from tools import start_debugger_on_exception
start_debugger_on_exception()

def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    #print('context',context)
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            nt=tokenizer.convert_ids_to_tokens(next_token)
            #print('nt',nt)
            if nt[0]=='[CLS]' and _>3:# or nt=='[SEP]':
                break
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    generated=generated.tolist()[0]
    context=context.tolist()[0]
    return generated[len(context):]
    #return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)


def main():
    #python interact2.py --model_path ./runs/gpt2-m-4.18 --length 200 --nsamples 1 --prefix 我常以“人就这么一辈子”这句话告诫自己并劝说朋友。这七个字，说来容易，听来简单，想起来却很深沉。它能使我在软弱时变得勇敢，骄傲时变得谦虚，颓废时变得积极，痛苦时变得欢愉，对任何事拿得起也放得下，所以我称它为“当头棒喝”、“七字箴言”。——我常想世间的劳苦愁烦、恩恩怨怨，如有不能化解的，不能消受的，不也就过这短短的几十年就烟消云散了吗？若是如此，又有什么解不开的呢？人就这么一辈子，想到了这句话，如果我是英雄，便要创造更伟大的功业；如果我是学者，便要获取更高的学问；如果我爱什么人，便要大胆地告诉她。因为今日过去便不再来了；这一辈子过去，便什么都消逝了。一本书未读，一句话未讲，便再也没有机会了。
    #python interact2.py --model_path ./runs/gpt2-m-1 --length 200 --nsamples 1 --prefix 唐僧高沙弥去参见药山禅师，药山禅师问：“你可知道人心像长安城一样热闹，熙熙攘攘的吗？”高沙弥说：“我的心中国泰民安。”药山问：“你的这种体悟是从读经得来的呢？还是从请益参学中得来的？”高沙弥说：“既不是从看经得，也不是从参学得。”药山问：“有人不看经也不参学，为什么得不到它呢？”高沙弥说：“不是他体悟不到，而是他不肯承担。”不开心不旷达的人，最重要的关键在于不能承担。不能承担就表示没有能力，能够承担的人，不管他遇到什么变局，是好是坏，是逆是顺，他都能定中生慧，凡事迎刃而解。也许这就是儒家所谓“造次必于是，颠沛必于是”的仁心至理。肯担当的人，便是对自己负责的人。惟能担当，便没有埋怨，没有忌妒，心中常保悠游自在。活力自然日增，生活也能活泼不拘。放下后的那种清爽，那种豪迈不拘的放旷，正是现代人所缺乏的。
    #python interact.py --model_path ./runs/gpt2-zh-2 --device '1' --length=400 --prefix 我常以“人就这么一辈子”这句话告诫自己并劝说朋友。这七个字，说来容易，听来简单，想起来却很深沉。
    #python interact.py --model_path ./runs/gpt2-zh-pt5-512-m2s5 --device '0' --length=1000 --prefix 有一个老总问他的下属：“你知道毛毛虫是怎么过河的吗？”下属说：“从桥上过。”老总摇摇头说：“没有桥。”下属说：“从叶子上过。”老总说：“叶子被水冲走了。”下属又接着说：“被鸟吃到肚子里就过河了。”老总笑说：“那样的话，毛毛虫就死掉了，也便失去了过河的意义。”那么毛毛虫究竟是怎么過河的呢？老总说：“毛毛虫要想过河，只有一种方法，那就是破茧成蝶。而这要经历一个痛苦的阶段才能实现蜕变。”
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=1, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=0.7, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=40, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='萧炎', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    r'''if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert'''
    #from transformers import BertTokenizer as tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # 此处设置程序使用哪些显卡
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device='cpu'
    #tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    tokenizer = BertTokenizer(vocab_file=args.model_path+'/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    print('params:',params_count(model))
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
    while True:
        #raw_text = args.prefix
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        #print('rawtext',raw_text)
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
        generated = 0
        for _ in range(nsamples // batch_size):
            out = generate(
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=length-len(raw_text)-2,
                is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
            )
            for i in range(batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                print(info)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                if args.save_samples:
                    samples_file.write(info)
                    samples_file.write(text)
                    samples_file.write('\n')
                    samples_file.write('=' * 90)
                    samples_file.write('\n' * 2)
        print("=" * 80)
        r'''if generated == nsamples:
            # close file when finish writing.
            if args.save_samples:
                samples_file.close()
            break'''

def infer():
    #nohup python -u interact.py --num 5 --device '6' --model_path ./runs/gpt2-zh-pt5-1024-s2m5 --data_path ./gpt2-zh/storal_zh_s2m.json > infer-gpt2-zh.out 2>&1 &
    #nohup python -u interact.py --num 5 --device '1' --model_path ./runs/gpt2-zh-pt5-1024-m2s5 --data_path ./zh_moral2story/zh_moral2story.json > infer-gpt2-zh2.log 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--data_path', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=1, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=0.7, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=40, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='萧炎', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--num', default=1, type=int, required=False)
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    r'''if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert'''
    #from transformers import BertTokenizer as tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'  # 此处设置程序使用哪些显卡
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device='cpu'
    #tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    tokenizer = BertTokenizer(vocab_file=args.model_path+'/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')

    with open('/home/liuziqi/storal/data/gpt2-zh/.'+args.data_path,'r',encoding='utf-8') as fout:
        data=json.load(fout)
    #with open('/home/liuziqi/storal/data/zh_moral2story/zh_moral2story.json'+args.data_path,'r',encoding='utf-8') as fout:
    #    data=json.load(fout)
    num=len(data['test'])
    print('datalen',num)
    output=[]
    for cnt in range(args.num):
        #f=open('/home/liuziqi/storal/generate/generate_gpt2_zh_s2m%d.txt'%(cnt+1),'w',encoding='utf-8')
        #f=open('/home/liuziqi/storal/generate/generate_gpt2_zh_m2s%d.txt'%(cnt+1),'w',encoding='utf-8')
        outdata=[]
        print('----------------%d--------------'%(cnt+1))
        tot=0
        for i in data['test']:
            tot+=1
            print(str(tot)+'/'+str(num))
            inputs=i[0]
            targets=i[1] 
            context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))
            out = generate(
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=1010-len(inputs),
                is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
            )
            text = tokenizer.convert_ids_to_tokens(out)
            for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                if is_word(item) and is_word(text[i + 1]):
                    text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'
            #info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
            #print(info)
            text = ''.join(text).replace('##', '').strip()
            #f.write(text+'\n')
            outdata.append(text)
        with open('/home/liuziqi/storal/generate/generate_gpt2_zh_m2s%d.json'%(cnt+1),'w',encoding='utf-8') as fout:
            json.dump(outdata,fout,indent=4,ensure_ascii=False)
        #f.close()
    #with open('/home/liuziqi/storal/generate/generate_gpt2_zh_s2m.json','w',encoding='utf-8') as fout:
    #    json.dump(output,fout,indent=4,ensure_ascii=False)
        

if __name__ == '__main__':
    main()
    #infer()
