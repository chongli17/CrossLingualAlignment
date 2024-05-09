import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Union, List
import re

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoModel, GPTNeoPreTrainedModel, GPTNeoForCausalLM
from transformers.models.xglm.modeling_xglm import XGLMModel, XGLMPreTrainedModel, XGLMForCausalLM
from transformers.models.bloom.modeling_bloom import BloomModel, BloomPreTrainedModel, BloomForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)


def iterate_in(encoder, input_ids, attention_mask, token_type_ids=None, output_hidden_states=False):
    # Interative input
    batch_size, num_sent, sent_len = input_ids.size()
    outs = {"last_hidden_state":[], "hidden_states":[]}
    outputs = None
    for s_i in range(num_sent):
        in_i = input_ids[:, s_i, :]
        att_i = attention_mask[:, s_i, :]
        tok_i = None
        if token_type_ids is not None:
            tok_i = token_type_ids[:, s_i, :]        
        outputs = encoder(
            in_i,
            attention_mask=att_i,
            token_type_ids=tok_i,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        outs["last_hidden_state"].append(outputs.last_hidden_state)
        if outputs.hidden_states is not None:
            outs["hidden_states"].append(outputs.hidden_states)
    outputs.last_hidden_state = torch.cat(outs["last_hidden_state"], dim=0).view(num_sent, batch_size, sent_len, -1).permute(1,0,2,3).reshape(batch_size*num_sent, sent_len, -1)
    if outputs.hidden_states is not None:
        hids = ()
        for h_i in range(len(outputs.hidden_states)):
            hids = hids + (torch.cat([out_i[h_i] for out_i in outs["hidden_states"]], dim=0).view(num_sent, batch_size, sent_len, -1).permute(1,0,2,3).reshape(batch_size*num_sent, sent_len, -1),)
    return outputs

def flatten_in(encoder, input_ids, attention_mask, token_type_ids=None, output_hidden_states=False):
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    return outputs


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_bottom': average of the bottom layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_bottom2': average of the first two layers.
    'avg_first_last': average of the first and the last layers.
    'last': the hidden state of the last token in the last layers.
    'last_top2': average of the hidden state of the last token in the last two layers.
    'last_first_last': average of the hidden state of the last token in the first and last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        self.pooler_re = re.compile(r'^(avg|last|max)-[0-9]+$')
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all",
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

    def need_out_hiddens(self, pooler_type):
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all",
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

        return pooler_type in [
            'avg_top2', 'avg_first_last', 
            "avg_bottom", "avg_bottom2", "avg_all",
            'last_top2', 'last_first_last'] or self.pooler_re.match(pooler_type)
        

    def average_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def max_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return torch.max((hidden_states * attention_mask.unsqueeze(-1)), dim=1)[0]

    def last_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        sequence_lengths = attention_mask.sum(-1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return self.average_embed(last_hidden, attention_mask)
        elif "avg-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.average_embed(hidden_states[l_idx], attention_mask)
        elif "max-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.max_embed(hidden_states[l_idx], attention_mask)
        elif "last-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.last_embed(hidden_states[l_idx], attention_mask)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            return self.average_embed((first_hidden + last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_bottom":
            first_hidden = hidden_states[1]
            return self.average_embed(first_hidden, attention_mask)
        elif self.pooler_type == "avg_bottom2":
            first_hidden = hidden_states[1]
            second_hidden = hidden_states[2]
            return self.average_embed((first_hidden + second_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            return self.average_embed((last_hidden + second_last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_all":
            num_layer = len(hidden_states)-1
            hidden_sum = 0
            for l in range(num_layer):
                hidden_sum += hidden_states[l+1]
            return self.average_embed(hidden_sum / float(num_layer), attention_mask)
        elif self.pooler_type == "last":
            return self.last_embed(last_hidden, attention_mask)
        elif "last-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.last_embed(hidden_states[l_idx], attention_mask)
        elif self.pooler_type == "last_first_last":
            first_layer_last = self.last_embed(hidden_states[1], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (first_layer_last + last_layer_last) / 2.0
        elif self.pooler_type == "last_top2":
            second_last_layer_last = self.last_embed(hidden_states[-2], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (second_last_layer_last + last_layer_last) / 2.0
        else:
            raise NotImplementedError

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls" or cls.model_args.cl_mlp:
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    clm_input_ids=None,
    clm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    clm_outputs = None

    # outputs = iterate_in(encoder, input_ids, attention_mask, token_type_ids, output_hidden_states=cls.pooler.need_out_hiddens(cls.model_args.pooler_type))
    # outputs = flatten_in(encoder, input_ids, attention_mask, token_type_ids, output_hidden_states=cls.pooler.need_out_hiddens(cls.model_args.pooler_type))

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
    
    encoder_input_dict={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # "head_mask":head_mask,
        "inputs_embeds":inputs_embeds,
        "output_attentions":output_attentions,
        "output_hidden_states": cls.pooler.need_out_hiddens(cls.model_args.pooler_type),
        "return_dict":True,
    }

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        encoder_input_dict["token_type_ids"] = token_type_ids
    
    if position_ids is not None:
        encoder_input_dict["position_ids"] = position_ids

    # Get raw embeddings
    outputs = encoder(
        **encoder_input_dict
    )
    
    # CLM auxiliary objective
    if clm_input_ids is not None:
        clm_input_ids = clm_input_ids.view((-1, clm_input_ids.size(-1)))
        # print(clm_input_ids.shape)

        clm_input_dict={
            "input_ids": clm_input_ids,
            # "attention_mask": attention_mask,
            # "head_mask":head_mask,
            # "inputs_embeds":inputs_embeds,
            # "output_attentions":output_attentions,
            # "output_hidden_states": cls.pooler.need_out_hiddens(cls.model_args.pooler_type),
            "return_dict":True,
        }

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        #     clm_input_dict["token_type_ids"] = token_type_ids
    
        # if position_ids is not None:
        #     clm_input_dict["position_ids"] = position_ids

        clm_outputs = encoder(
            **clm_input_dict
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls" or cls.model_args.cl_mlp:
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # Calculate the similariy of z1.unsqueeze(1) (bs, 1, hidden) and z2.unsqueeze(0) (1, bs, hidden)
    # Output: cos_sim (bs, bs)
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        # construct a weight matrix like:
        # [[0.0000, 0.0000, 0.0000, z3_weight, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, z3_weight, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, z3_weight]]
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for CLM
    if clm_outputs is not None and clm_labels is not None:
        clm_labels = clm_labels.view(-1, clm_labels.size(-1))
        lm_logits = cls.lm_head(clm_outputs.last_hidden_state)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = clm_labels[..., 1:].contiguous()

        clm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss = cls.model_args.cl_weight*loss + cls.model_args.clm_weight * clm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    encoder_input_dict={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "head_mask":head_mask,
        "inputs_embeds":inputs_embeds,
        "output_attentions":output_attentions,
        "output_hidden_states": cls.pooler.need_out_hiddens(cls.model_args.pooler_type),
        "return_dict":True,
    }

    if token_type_ids is not None:
        encoder_input_dict["token_type_ids"] = token_type_ids
    
    if position_ids is not None:
        encoder_input_dict["position_ids"] = position_ids

    outputs = encoder(
        **encoder_input_dict
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class GPT2ForCL(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.transformer = GPT2Model(config)
        self.lm = self.transformer

        if self.model_args.do_clm:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        cl_init(self, config)

    def clm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
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
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        clm_input_ids=None,
        clm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clm_input_ids=clm_input_ids,
                clm_labels=clm_labels,
            )

class GPTNeoForCL(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.transformer = GPTNeoModel(config)
        self.lm = self.transformer

        if self.model_args.do_clm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        cl_init(self, config)
    
    def clm_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        clm_input_ids=None,
        clm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clm_input_ids=clm_input_ids,
                clm_labels=clm_labels,
            )

class XGLMForCL(XGLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = XGLMModel(config)
        self.lm = self.model

        if self.model_args.do_clm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        cl_init(self, config)
    
    def clm_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # shift labels and add a pad token to the end
            shift_labels = labels.new_zeros(labels.shape)
            shift_labels[:, :-1] = labels[:, 1:].clone()
            shift_labels[:, -1] = self.config.pad_token_id

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None, #xglm doesnot have token_type_ids
        # position_ids=None, #xglm doesnot have position_ids
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        clm_input_ids=None,
        clm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clm_input_ids=clm_input_ids,
                clm_labels=clm_labels,
            )

class BloomForCL(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.transformer = BloomModel(config)
        self.lm = self.transformer

        if self.model_args.do_clm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        cl_init(self, config)
    
    def clm_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None, #bloom doesnot have token_type_ids
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        clm_input_ids=None,
        clm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clm_input_ids=clm_input_ids,
                clm_labels=clm_labels,
            )

class LlamaForCL(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = LlamaModel(config)
        self.lm = self.model

        if self.model_args.do_clm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        cl_init(self, config)
    
    def clm_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None, #llama doesnot have token_type_ids
        position_ids=None,
        # head_mask=None, #llama doesnot have head mask
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        clm_input_ids=None,
        clm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clm_input_ids=clm_input_ids,
                clm_labels=clm_labels,
            )