import torch
from torch import nn
import torch.nn.functional as F
import math

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
                 fc_dim=128,
                 hidden_dim=128,
                 num_heads=2,
                 drop_out=0.15,
                 out_dim=8,
                 mode = 'positive'
                 ):
        super().__init__()

        # text bs 20 300
        # visual bs 20 35
        # acoustic bs 20 74
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.dim = fc_dim
        self.mode = mode

        self.fc1 = nn.Linear(self.text_dim, fc_dim)
        self.fc2 = nn.Linear(self.visual_dim, fc_dim)
        self.fc3 = nn.Linear(self.audio_dim, fc_dim)

        self.crossattention_text_visual = CrossAttention(
            fc_dim=fc_dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        self.crossattention_text_audio = CrossAttention(
            fc_dim=fc_dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        self.crossattention_visual_audio = CrossAttention(
            fc_dim=fc_dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )
        
        if mode != 'all':
            self.gru1 = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru2 = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru3 = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.fc = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out), 
                            nn.BatchNorm1d(self.dim * 2),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, out_dim))

        else:
            self.gru1_positive = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru2_positive = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru3_positive = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru1_negative = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru2_negative = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
            self.gru3_negative = nn.GRU(self.dim, self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)


            self.fc_weight = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, 1),
                            nn.Sigmoid())

            self.fc_positive = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2),
                            nn.ReLU6(),
                            nn.Linear(self.dim * 2, out_dim))
            self.fc_negative = nn.Sequential(
                            nn.Linear(self.dim * 3, self.dim * 2),
                            nn.Dropout(drop_out * 2), 
                            nn.BatchNorm1d(self.dim * 2),
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
                fc_dim=128,
                hidden_dim=128,
                num_heads=4,
                drop_out=0.15,
                gru_hiddendim=128,
                out_dim=8
                ):
        super().__init__()
        self.gru_hiddendim = gru_hiddendim
        self.fc_text = nn.Linear(text_dim, hidden_dim)
        self.fc_visual = nn.Linear(visual_dim, hidden_dim)
        self.fc_audio = nn.Linear(audio_dim, hidden_dim)

        self.gru_text = nn.GRU(hidden_dim , gru_hiddendim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.gru_visual = nn.GRU(hidden_dim , gru_hiddendim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.gru_audio = nn.GRU(hidden_dim , gru_hiddendim, 1, batch_first=True, bidirectional=True, dropout=0.0)

        self.fc = nn.Sequential(
                        nn.Linear(gru_hiddendim * 3, gru_hiddendim * 2),
                        nn.Dropout(drop_out), 
                        nn.BatchNorm1d(gru_hiddendim * 2),
                        nn.ReLU6(),
                        nn.Linear(gru_hiddendim * 2, out_dim)
        )
    def forward(self,text_features,visual_features,audio_features):

        text_features, _ = self.gru_text(self.fc_text(text_features))
        visual_features, _ = self.gru_visual(self.fc_visual(visual_features))
        audio_features, _ = self.gru_audio(self.fc_audio(audio_features))

        out = torch.cat((text_features, visual_features, audio_features), -1)

        out = torch.mean(out[:, :, self.gru_hiddendim * 3:] + out[:, :, :self.gru_hiddendim * 3], 1)
        out = self.fc(out)

        return out


class Model(nn.Module):
    def __init__(self, args ):
        super().__init__()
        self.net = CrossMdoalBlock(mode = args.mode)
        # self.net = baseline()
    def forward(self,
                text_features,
                visual_features,
                audio_features):
        
        out = self.net(
                text_features,
                visual_features,
                audio_features
                )
        
        return out





if __name__ == '__main__':
# Example usage
    text_features = torch.randn(16, 20, 300)
    speech_features = torch.randn(16, 20, 35)
    audi_features = torch.randn(16, 20, 74)
    
    cross_attention = CrossMdoalBlock()
    out = cross_attention(text_features, speech_features,audi_features)
    print(out.shape)  # Output shapes: torch.Size([16, 20, 128]), torch.Size([16, 20, 128])