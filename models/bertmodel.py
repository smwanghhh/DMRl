
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler


from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from einops import rearrange, repeat
from torch.nn import functional as F
import math

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

class MultimodalConfig(object):
    def __init__(self, dropout_prob, emb_size, dim_t, dim_a, dim_v, seqlength, mode):
        self.dropout_prob = dropout_prob
        self.emb_size = emb_size
        self.dim_t = dim_t
        self.dim_a = dim_a
        self.dim_v = dim_v
        self.seqlength = seqlength
        self.mode = mode

class BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)  ####get word embedding form bert
        self.encoder = BertEncoder(config)
        self.drop_out = multimodal_config.dropout_prob
        self.emb_size = multimodal_config.emb_size
        self.length = multimodal_config.seqlength
        self.dim_t, self.dim_v, self.dim_a = multimodal_config.dim_t, multimodal_config.dim_v, multimodal_config.dim_a
        ##define your model

        self.attention_model = CrossMdoalBlock(
                    text_dim=self.dim_t,
                    visual_dim=self.dim_v,
                    audio_dim=self.dim_a,
                    out_dim=1,
                    dim = self.emb_size,
                    drop_out=self.drop_out,
                    mode = multimodal_config.mode,
                    )



    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising variou           s elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        batch = visual.shape[0]
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        ###transformer
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        text = encoder_outputs[0]
        # print(text.shape)
        # print(visual.shape)
        # print(acoustic.shape)
        ###you have got text visual acoustic three modalities
        ###feedforward your model

        out = self.attention_model(text,visual,acoustic)
        out = torch.unsqueeze(out,1)
        return out
    
class MultiheadAttention1(nn.Module):
    def __init__(self, input_dim, heads, dropout):
        super(MultiheadAttention1, self).__init__()

        self.num_attention_heads = heads
        self.attention_head_size = input_dim // heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)
        self.dense1 = nn.Linear(input_dim, input_dim)

        self.norm = nn.LayerNorm(input_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input1, input2, input3):
        mixed_query_layer = self.query(input1)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input2)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input3)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        out = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]
        # out = context_layer + self.dense1(context_layer)
        return self.norm(self.dense1(out))


class MultiheadAttention2(nn.Module):
    def __init__(self, input_dim, heads, dropout):
        super(MultiheadAttention2, self).__init__()

        self.num_attention_heads = heads
        self.attention_head_size = input_dim // heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)
        self.dense1 = nn.Linear(input_dim, input_dim)

        self.norm = nn.LayerNorm(input_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input1, input2, input3):
        mixed_query_layer = self.query(input1)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input2)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input3)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        
        attention_probs = self.dropout(torch.ones_like(attention_scores) - nn.Softmax(dim=-1)(attention_scores))
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        out = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]
        # out = context_layer + self.dense1(context_layer)
        return self.norm(self.dense1(out))


    
class CrossAttention(nn.Module):
    def __init__(self, fc_dim, num_heads, dropout, mode):
        super(CrossAttention, self).__init__()

        self.mode = mode

        if mode == 'positive':
            self.a_attention = MultiheadAttention1(fc_dim, num_heads, dropout)
            self.b_attention = MultiheadAttention1(fc_dim, num_heads, dropout)
        elif mode == 'negative':
            self.a_attention = MultiheadAttention2(fc_dim, num_heads, dropout)
            self.b_attention = MultiheadAttention2(fc_dim, num_heads, dropout)
        else:
            self.a_attention_positive = MultiheadAttention1(fc_dim, num_heads, dropout)
            self.b_attention_positive = MultiheadAttention1(fc_dim, num_heads, dropout)
            self.a_attention_negative = MultiheadAttention2(fc_dim, num_heads, dropout)
            self.b_attention_negative = MultiheadAttention2(fc_dim, num_heads, dropout)

    def forward(self, a_features, b_features):

        # Cross-attention between text and speech features
        if self.mode != 'all':
            b_out = self.a_attention(a_features, b_features, b_features)
            a_out = self.b_attention(b_features, a_features, a_features)
        else:
            b_out_positive = self.a_attention_positive(a_features, b_features, b_features)
            b_out_negative = self.a_attention_negative(a_features, b_features, b_features)
            a_out_positive = self.b_attention_positive(b_features, a_features, a_features)
            a_out_negative = self.b_attention_negative(b_features, a_features, a_features)
            a_out = [a_out_positive, a_out_negative]
            b_out = [b_out_positive, b_out_negative]
        return a_out, b_out


class CrossMdoalBlock(nn.Module):
    def __init__(self,
                 text_dim=300,
                 visual_dim=35,
                 audio_dim=74,
                 dim=128,
                 drop_out=0.15,
                 out_dim=1,
                 num_heads = 2,
                 mode = 'positive'
                 ):
        super().__init__()

        # text bs 20 300
        # visual bs 20 35
        # acoustic bs 20 74
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.dim = dim
        self.mode = mode

        self.fc1 = nn.Linear(self.text_dim, dim)
        self.fc2 = nn.Linear(self.visual_dim, dim)
        self.fc3 = nn.Linear(self.audio_dim, dim)

        self.crossattention_text_visual = CrossAttention(
            fc_dim=dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        self.crossattention_text_audio = CrossAttention(
            fc_dim=dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        self.crossattention_visual_audio = CrossAttention(
            fc_dim=dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        
        if mode != 'all':
            self.gru1 = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)
            self.gru2 = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)
            self.gru3 = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)
            self.fc = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 3), 
                            nn.BatchNorm1d(self.dim * 2, eps=1e-5),
                            nn.Linear(self.dim * 2, out_dim))

        else:
            self.gru1_positive = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru2_positive = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru3_positive = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru1_negative = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru2_negative = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru3_negative = nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.0)


            self.fc_weight = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2, eps=1e-5),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, 1),
                            nn.Sigmoid())

            self.fc_positive = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2, eps=1e-5),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, out_dim))
            self.fc_negative = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2, eps=1e-5),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, out_dim))

    def forward(self,
                text_features,
                visual_features,
                audio_features
                ):
        text_features = self.fc1(text_features)
        visual_features = self.fc2(visual_features)
        audio_features = self.fc3(audio_features)

        text_out1, audio_out1 = self.crossattention_text_audio(
                                text_features,audio_features
                                )
        
        text_out2, visual_out1 = self.crossattention_text_visual(
                                text_features,visual_features
                                )
        
        visual_out2, audio_out2 = self.crossattention_visual_audio(
                                visual_features,audio_features
                                )
        if self.mode != 'all':
            text_out = (text_out1 + text_out2)/ 2 # torch.cat((text_out1, text_out2), -1)
            visual_out = (visual_out1 + visual_out2) / 2#torch.cat((visual_out1, visual_out2), -1)
            audio_out = (audio_out1 + audio_out2)/ 2 #torch.cat((audio_out1, audio_out2), -1)
       
            text_out,  _ = self.gru1(text_out)
            visual_out, _ = self.gru2(visual_out)
            audio_out, _ = self.gru3(audio_out)
            out = torch.cat((text_out, visual_out, audio_out), -1)
            out = torch.mean((out[:, :, self.dim*3:] + out[:, :, :self.dim*3])/2, 1)
            out = self.fc(out)
        else:
            text_out_positive = (text_out1[0] + text_out2[0])/ 2
            text_out_negative = (text_out1[1] + text_out2[1])/ 2
            visual_out_positive = (visual_out1[0] + visual_out2[0])/ 2
            visual_out_negative = (visual_out1[1] + visual_out2[1])/ 2
            audio_out_positive = (audio_out1[0] + audio_out2[0])/ 2
            audio_out_negative = (audio_out1[1] + audio_out2[1])/ 2
            text_out_positive,  _ = self.gru1_positive(text_out_positive)
            visual_out_positive, _ = self.gru2_positive(visual_out_positive)
            audio_out_positive, _ = self.gru3_positive(audio_out_positive)
            text_out_negative,  _ = self.gru1_negative(text_out_negative)
            visual_out_negative, _ = self.gru2_negative(visual_out_negative)
            audio_out_negative, _ = self.gru3_negative(audio_out_negative)


            out_positive = torch.cat((text_out_positive, visual_out_positive, audio_out_positive), -1)
            out_positive = torch.mean((out_positive[:, :, self.dim*3:] + out_positive[:, :, :self.dim*3])/2, 1)
            positive_emb = out_positive
            out_positive = self.fc_positive(out_positive)

            out_negative = torch.cat((text_out_negative, visual_out_negative, audio_out_negative), -1)
            out_negative = torch.mean((out_negative[:, :, self.dim*3:] + out_negative[:, :, :self.dim*3])/2, 1)
            negative_emb = out_negative
            out_negative = self.fc_negative(out_negative)

            weight = self.fc_weight(positive_emb + negative_emb)

            out = weight * out_positive + (1-weight) * out_negative
        return out

class baseline(nn.Module):
    def __init__(self,
                text_dim=300,
                visual_dim=35,
                audio_dim=74,
                dim=128,
                drop_out=0.15,
                out_dim=1
                ):
        super().__init__()
        self.dim = dim
        self.fc_text = nn.Linear(text_dim, self.dim)
        self.fc_visual = nn.Linear(visual_dim, self.dim)
        self.fc_audio = nn.Linear(audio_dim, self.dim)

        self.gru_text =   nn.GRU(self.dim, self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)
        self.gru_visual = nn.GRU(self.dim , self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)
        self.gru_audio =  nn.GRU(self.dim , self.dim, 3, batch_first=True, bidirectional=True, dropout=0.)

        self.fc = nn.Sequential(
                        nn.Linear(self.dim * 3, self.dim * 2),
                        nn.Dropout(drop_out), 
                        nn.BatchNorm1d(self.dim * 2, eps=1e-5),
                        # nn.ReLU6(),
                        nn.Linear(self.dim * 2, out_dim)
        )
    def forward(self,text_features,visual_features,audio_features):

        text_features, _ = self.gru_text(self.fc_text(text_features))
        visual_features, _ = self.gru_visual(self.fc_visual(visual_features))
        audio_features, _ = self.gru_audio(self.fc_audio(audio_features))

        out = torch.cat((text_features, visual_features, audio_features), -1)

        out = torch.mean((out[:, :, self.dim * 3:] + out[:, :, :self.dim * 3])/2, 1)
        out = self.fc(out)

        return out
    

if __name__ == '__main__':


    bert_model = "bert-base-uncased"

    multimodal_config = MultimodalConfig(
                    dropout_prob=0.0, 
                    emb_size=256, 
                    dim_t=768,
                    dim_a=74, 
                    dim_v=47, 
                    seqlength= 50
                    )    
    model = BertModel.from_pretrained(
                        bert_model, 
                        multimodal_config=multimodal_config, 
                        num_labels=1)