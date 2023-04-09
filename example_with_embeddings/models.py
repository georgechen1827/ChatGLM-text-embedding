import warnings
from types import MethodType
from typing import Optional, Tuple

from transformers import AutoModel,AutoTokenizer
from peft import PeftModel, PrefixTuningConfig, TaskType, get_peft_model, PromptLearningConfig, PeftType
import torch

MODEL = "THUDM/chatglm-6b-int4" #"C:\Documents\data\大型语言模型\Models\ChatGLM-6B-main\model-int4"
Embedding_prefix = "./output/checkpoint-1000/adapter_model.bin"

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

def peft_forward(
    self,
    input_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    batch_size = input_ids.shape[0]
    if attention_mask is not None:
        # concat prompt attention mask
        prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
        warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
        kwargs["token_type_ids"] = None
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    past_key_values = self.get_prompt(batch_size)
    return embedding_forward(self.base_model,input_ids=input_ids, past_key_values=past_key_values, **kwargs) # self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,truncation_side="left")
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).half().cuda()
model = model.eval()

# setup peft
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prefix_projection=False
)

embedding_model = get_peft_model(model, peft_config).half().cuda()

embedding_model.load_state_dict(torch.load(Embedding_prefix),strict=False)

embedding_model.forward = MethodType(peft_forward,embedding_model)