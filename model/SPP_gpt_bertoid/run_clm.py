# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import torch
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from tools import start_debugger_on_exception
start_debugger_on_exception()
from datasets import load_dataset

import transformers
from transformers import (
    GPT2LMHeadModel,
    GPT2Model,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch.nn as nn

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# st======================================================================================================================================

class MyModel(GPT2Model):
    '''
    def from_pretrained(self, config, model_args, tokenizer):
        self.tokenizer = tokenizer
        self.sp_id = tokenizer.convert_tokens_to_ids("[P]")
        self.label_id = tokenizer.convert_tokens_to_ids("[S]")

        self.model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        # )

        self.model.resize_token_embeddings(len(self.tokenizer))
        # print(len(self.tokenizer))
        self.num_labels = config.num_labels
        self.loss = nn.CrossEntropyLoss()
    def __init__(self):
        super().__init__()'''
    
    def __init__(self, config, model_name_or_path, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.sp_id = tokenizer.convert_tokens_to_ids("[P]")
        self.label_id = tokenizer.convert_tokens_to_ids("[S]")

        self.model = GPT2Model.from_pretrained(model_name_or_path)
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        # )

        self.model.resize_token_embeddings(len(self.tokenizer))
        # print(len(self.tokenizer))
        self.num_labels = config.num_labels
        self.loss = nn.CrossEntropyLoss()
        #self.state_dict = 
        # print(type(self.base_model))
        # logging.info(self.base_model)

    def forward(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,                
        labels = None,
    ): # input: text token
        # get hidden states from token_index
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        '''
        print(len(output))
        if output.logits is not None:
            print(f'logits type = {type(output.logits)}, logits shape ={output.logits.shape}')
        if output.loss is not None:
            print(f'loss type = {type(output.loss)}, loss shape = {type(output.loss)}, {output.loss}')
        if output.hidden_states is not None:
            print(f'hidden states type = {type(output.hidden_states)}')
        if output.attentions is not None:
            print(f'attention shape = {output.attention.shape}')
        '''
        # positions = (input_ids == self.sp_id).float() # [4, 11]
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,            
        )

        # [batch_size, length, hidden_size]
        encoder_hidden_states = output.last_hidden_state #outputs["encoder_last_hidden_state"]
        # [batch_size, length]
        mask1 = torch.eq(input_ids, torch.tensor(self.sp_id).to(input_ids.device)).float()
        mask2 = torch.eq(input_ids, torch.tensor(self.tokenizer.mask_token_id).to(input_ids.device)).float()
        # [batch_size, length]
        logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
        logits -= (1 - mask2) * (1e20)
        # print(logits[0])
        lprobs = torch.nn.functional.softmax(logits, dim=-1)
        # print(lprobs[0])
        #import pdb;pdb.set_trace()
        labels = torch.sum(torch.cat([torch.eq(input_ids, self.label_id).int()[:, 1:], torch.zeros(input_ids.size()[0], 1).int().to('cuda:5')], 1) * input_ids, 1)
        # [batch_size]
        # label = tgt_ids[:, 0]
        # [batch_size, length]
        mask3 = torch.eq(torch.cumsum(torch.eq(torch.cumsum(mask2, 1), labels[:, None]).float(), 1), 1).float()
        loss = -torch.mean(torch.log(torch.sum(lprobs*mask3, 1)+1e-20))

        return transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def save_pretrained(self, save_directory, state_dict = None):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

# ed======================================================================================================================================
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #training_args.device = torch.device(type='cuda', index=3)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer.add_special_tokens({"additional_special_tokens": ["[P]", "[S]"]})

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        #from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #config=config,
        #cache_dir=model_args.cache_dir,
        #revision=model_args.model_revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    )
    model = MyModel(config=config, model_name_or_path = model_args.model_name_or_path, tokenizer=tokenizer)
    #.from_pretrained(config=config, model_args=model_args, tokenizer=tokenizer,
    #                pretrained_model_name_or_path=model_args.model_name_or_path,)

    model.resize_token_embeddings(len(tokenizer))
    num_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.size(), torch.numel(param))
            num_param += torch.numel(param)
    print("="*10)
    print("# Parameters:", num_param)
    print("="*10)




    # st======================================================================================================================================
    datasets = {}
    with open(data_args.train_file) as fin:
        datasets["train"] = [line.strip() for line in fin]
    with open(data_args.validation_file) as fin:
        datasets["validation"] = [line.strip() for line in fin]

    class customdataset(torch.utils.data.Dataset):
        def __init__(self, encodings_labels_dict):
            self.encodings = encodings_labels_dict["input_ids"]
            self.labels = encodings_labels_dict["labels"]

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer.pad_token = "[PAD]"
    def preprocess_function(examples):
        result = {}
        result["input_ids"] = tokenizer([l for l in examples], truncation=True, padding=True, max_length=300)
        result["labels"] = [ids[:-1]+[tokenizer.pad_token_id] for ids in result["input_ids"].input_ids]
        return result

    train_dataset = customdataset(preprocess_function(datasets["train"]))
    eval_dataset = customdataset(preprocess_function(datasets["validation"]))
    # ed======================================================================================================================================

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,#lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset,#lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_clm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
