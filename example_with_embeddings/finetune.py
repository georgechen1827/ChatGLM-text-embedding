# from transformers.integrations import TensorBoardCallback
# from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple

from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, TaskType, PrefixTuningConfig, LoraConfig
from dataclasses import dataclass, field
import datasets
import os

@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="medi-data/processed")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)

from types import MethodType
def embedding_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    MASK, gMASK = 150000, 150001
    mask_token = MASK if MASK in input_ids else gMASK
    use_gmask = False if MASK in input_ids else gMASK
    seqs = input_ids.tolist()
    mask_positions = [seq.index(mask_token) for seq in seqs]

    position_ids = self.get_position_ids(
        input_ids,
        mask_positions = mask_positions,
        device=input_ids.device,
    )

    past_key_values = [torch.permute(i,(0,3,1,2,4)) for i in past_key_values]

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs.last_hidden_state[-1]

    return hidden_states


# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x):
#         return super().forward(x).to(torch.float32)


# def data_collator(features: list) -> dict:
#     len_ids = [len(feature["input_ids"]) for feature in features]
#     longest = max(len_ids)
#     input_ids = []
#     labels_list = []
#     for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
#         ids = feature["input_ids"]
#         seq_len = feature["seq_len"]
#         labels = (
#             [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
#         )
#         ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
#         _ids = torch.LongTensor(ids)
#         labels_list.append(torch.LongTensor(labels))
#         input_ids.append(_ids)
#     input_ids = torch.stack(input_ids)
#     labels = torch.stack(labels_list)
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#     }


class ModifiedTrainer(Trainer):
    cl_temperature = 0.01

    def compute_loss(self, model, inputs, return_outputs=False):
        # for task_id in inputs['task_name']:
        #     assert task_id==inputs['task_name'][0],f"Examples in the same batch should come from the same task, " \
        #                                          f"but task {task_id} and task {inputs['task_name'][0]} are found"
        cur_results = {}
        for k in ['query', 'pos', 'neg']:
            cur_inputs = {
                'input_ids': inputs[f'{k}_input_ids'],
                # 'attention_mask': inputs[f'{k}_attention_mask'],
                # 'context_masks': inputs[f'{k}_context_masks'],
            }
            cur_results[k] = model(**cur_inputs)
        embeddings_query = cur_results['query']
        embeddings_pos = cur_results['pos']
        embeddings_neg = cur_results['neg']

        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

ignore_pad_token_for_loss = True

from arguments import ModelArguments

def main():
    # writer = SummaryWriter()
    finetune_args, training_args,model_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments,ModelArguments)
    ).parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    # init model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, truncation_side="left")

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    ).half().cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    original_forward = model.forward
    model.forward = MethodType(embedding_forward, model)

    # setup peft
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=8,
        # encoder_hidden_size=8,
        prefix_projection=True
    )
    model = get_peft_model(model, peft_config)

    model = model.half().cuda()

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    print(f"\n{len(dataset)=}\n")

    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        # callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    # writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()