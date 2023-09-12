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

    def forward(self, input):
        mixed_query_layer = self.query(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input)  # [Batch_size x Seq_length x Hidden_size]

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

    def forward(self, input):
        mixed_query_layer = self.query(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input)  # [Batch_size x Seq_length x Hidden_size]

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
            self.attention = MultiheadAttention1(fc_dim, num_heads, dropout)
        else:
            self.attention = MultiheadAttention2(fc_dim, num_heads, dropout)


    def forward(self, input):
        # Cross-attention between text and speech features
        out = self.attention(input)
        return out

        
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

        self.crossattention = CrossAttention(
            fc_dim=fc_dim,
            num_heads=num_heads,
            dropout=drop_out,
            mode = mode
        )

        
        self.gru_text = nn.GRU(self.dim , self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.gru_visual = nn.GRU(self.dim , self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.gru_audio = nn.GRU(self.dim , self.dim, 1, batch_first=True, bidirectional=True, dropout=0.0)

        self.fc = nn.Sequential(
                        nn.Linear(self.dim * 3, self.dim * 2),
                        nn.Dropout(drop_out * 5), 
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

        input = torch.cat((text_features, visual_features, audio_features), 1)
        out = self.crossattention(input)

        text_features, visual_features, audio_features = out[:, :20], out[:,20:40], out[:, 40:]

        text_features, _ = self.gru_text(text_features)
        visual_features, _ = self.gru_visual(visual_features)
        audio_features, _ = self.gru_audio(audio_features)

        out = torch.cat((text_features, visual_features, audio_features), -1)

        out = torch.mean(out[:, :, self.dim * 3:] + out[:, :, :self.dim * 3], 1)
        out = self.fc(out)

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
        # self.net = baseline()
        self.net = CrossMdoalBlock(mode = args.mode)

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