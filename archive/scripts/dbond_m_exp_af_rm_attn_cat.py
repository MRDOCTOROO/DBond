import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from typing import List, Tuple
import yaml


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, forward_expansion, **kwargs):
        super(AttentionBlock, self).__init__()
        self.attentionLayer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            **kwargs
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, forward_expansion * hidden_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attn_output_batch, attn_weight_batch = self.attentionLayer.forward(
            query=query, key=key, value=value, **kwargs
        )
        x = self.layer_norm_1(query + self.dropout(attn_output_batch))
        res_x = x
        x = self.ffn(x)
        out = self.layer_norm_2(res_x + self.dropout(x))
        return out, attn_weight_batch


class Scalar2VecBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(Scalar2VecBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        # scalar: [batch,1]
        # vec: [batch, hidden_dim]
        vec = self.block(scalar)
        return vec


class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, dropout):
        super(StateEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList(
            [Scalar2VecBlock(hidden_dim, dropout) for _ in range(state_dim)]
        )

        self.embed_dim = hidden_dim

    def forward(self, state_vec_batch: torch.Tensor) -> torch.Tensor:
        # state_vec_batch: [batch_len,state_dim]
        state_vec_batch = state_vec_batch.float()
        cnt = 0
        #  x: [batch_len,state_dim,HIDDEN_DIM]
        x = torch.empty(
            (state_vec_batch.shape[0], state_vec_batch.shape[1], self.embed_dim),
            dtype=torch.float,
            device=state_vec_batch.device,
        )
        for scalar2vec in self.scalar2vec:
            x[:, cnt, :] = scalar2vec(state_vec_batch[:, cnt].unsqueeze(1))
            cnt += 1
        return x


class EnvEncoder(nn.Module):
    def __init__(self, env_dim, hidden_dim, dropout):
        super(EnvEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList(
            [Scalar2VecBlock(hidden_dim, dropout) for _ in range(env_dim)]
        )

        self.embed_dim = hidden_dim

    def forward(self, env_vec_batch: torch.Tensor) -> torch.Tensor:
        # env_vec_batch: [batch_len,ENV_DIM]

        env_vec_batch = env_vec_batch.float()
        cnt = 0
        #  x: [batch_len,env_dim,HIDDEN_DIM]
        x = torch.empty(
            (env_vec_batch.shape[0], env_vec_batch.shape[1], self.embed_dim),
            dtype=torch.float,
            device=env_vec_batch.device,
        )
        for scalar2vec in self.scalar2vec:
            x[:, cnt, :] = scalar2vec(env_vec_batch[:, cnt].unsqueeze(1))
            cnt += 1
        return x


class SeqEncoder(nn.Module):
    def __init__(
        self, hidden_dim, num_heads, dropout, forward_expansion, attention_layer_num
    ):
        super(SeqEncoder, self).__init__()
        self.self_attention = nn.ModuleList(
            [
                AttentionBlock(hidden_dim, num_heads, dropout, forward_expansion)
                for _ in range(attention_layer_num)
            ]
        )

    def forward(
        self, seq_embedding_batch: torch.Tensor, seq_padding_mask_batch: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # seq_padding_mask_batch : [batch_len,max_seq_len],true for pad pos
        attn_output_batch = seq_embedding_batch
        # attn_output_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # attn_output_vec_batch : [batch_len,1,HIDDEN_DIM]
        # attn_weight_batch : [batch_len,max_seq_len,max_seq_len]
        # attn_weight_batch_list : [attention_layer_num,batch_len,max_seq_len,max_seq_len]
        attn_weight_batch_list = []
        for layer in self.self_attention:
            attn_output_batch, attn_weight_batch = layer.forward(
                attn_output_batch,
                attn_output_batch,
                attn_output_batch,
                key_padding_mask=seq_padding_mask_batch,
            )
            attn_weight_batch_list.append(attn_weight_batch)
        # attn_output_vec_batch = self.masked_mean(attn_output_batch,~seq_padding_mask_batch)
        return attn_output_batch, attn_weight_batch_list


class Encoder(nn.Module):
    def __init__(
        self,
        aa_type_count,
        pad_index,
        state_dim,
        env_dim,
        hidden_dim,
        num_heads,
        dropout,
        forward_expansion,
        attention_layer_num,
    ):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(
            aa_type_count, hidden_dim, padding_idx=pad_index
        )
        self.pos_encoder = Summer(PositionalEncoding1D(hidden_dim))
        self.seq_self_encoder = SeqEncoder(
            hidden_dim, num_heads, dropout, forward_expansion, attention_layer_num
        )
        self.state_encoder = StateEncoder(state_dim, hidden_dim, dropout)
        self.env_encoder = EnvEncoder(env_dim, hidden_dim, dropout)

    def masked_mean(self, input: torch.Tensor, mask: torch.Tensor):
        # input: [batch,seq_len,hidden]
        # mask: [batch,seq_len], bool ,true is valid
        mask_float = mask.float()
        valid_counts = mask_float.sum(dim=1, keepdim=True)
        weighted_sum = (input * mask_float.unsqueeze(-1)).sum(dim=1, keepdim=True)
        mean = weighted_sum / valid_counts.clamp(min=1e-6).unsqueeze(-1)
        # mean: [batch,1,hidden]
        return mean

    def masked_tensor(self, input: torch.Tensor, mask: torch.Tensor):
        # input: [batch,seq_len,hidden]
        # mask: [batch,seq_len], bool ,true is valid
        mask_float = mask.float()
        return input * mask_float.unsqueeze(-1)

    def forward(
        self,
        seq_index_batch: torch.Tensor,
        seq_padding_mask_batch: torch.Tensor,
        state_vec_batch: torch.Tensor,
        env_vec_batch: torch.Tensor,
    ) -> torch.Tensor:
        # seq_index_batch : [batch_len,max_seq_len]
        # seq_padding_mask_batch : [batch_len,max_seq_len]

        # state_vec_batch: [batch_len,state_dim]
        # env_vec_batch: [batch_len,ENV_DIM]

        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        seq_embedding_batch: torch.Tensor = self.embedding_layer(seq_index_batch)
        seq_embedding_batch = self.pos_encoder(seq_embedding_batch)

        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # seq_attn_weight_list :[attention_layer_num,batch_len,max_seq_len,max_seq_len]
        seq_embedding_batch, seq_attn_weight_list = self.seq_self_encoder.forward(
            seq_embedding_batch, seq_padding_mask_batch
        )
        # seq_vec_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        seq_vec_batch = self.masked_tensor(seq_embedding_batch, ~seq_padding_mask_batch)

        # state_vec_batch: [batch_len,state_dim,HIDDEN_DIM]
        # env_vec_batch: [batch_len,env_dim,HIDDEN_DIM]
        env_vec_batch = self.env_encoder.forward(env_vec_batch)
        state_vec_batch = self.state_encoder.forward(state_vec_batch)
        # env_vec_batch_expand: [batch_len,max_seq_len,env_dim,HIDDEN_DIM]
        env_vec_batch_expand = env_vec_batch.unsqueeze(1).expand(
            -1, seq_vec_batch.shape[1], -1, -1
        )
        # state_vec_batch_expand: [batch_len,max_seq_len,state_dim,HIDDEN_DIM]
        state_vec_batch_expand = state_vec_batch.unsqueeze(1).expand(
            -1, seq_vec_batch.shape[1], -1, -1
        )
        # seq_vec_batch : [batch_len,max_seq_len,1,HIDDEN_DIM]
        seq_vec_batch = seq_vec_batch.unsqueeze(2)
        # latent_vec_batch : [batch_len,max_seq_len,1+state_dim+env_dim,HIDDEN_DIM]
        latent_vec_batch = torch.cat(
            (seq_vec_batch, state_vec_batch_expand, env_vec_batch_expand), dim=2
        )
        # latent_vec_batch : [batch_len,max_seq_len,(1+state_dim+env_dim)*HIDDEN_DIM]
        latent_vec_batch = latent_vec_batch.flatten(start_dim=2)
        return latent_vec_batch


# ...existing code...


class FeatureFusionLayer(nn.Module):
    def __init__(
        self,
        state_dim,
        env_dim,
        hidden_dim,
        output_dim,
        dropout,
        num_heads=4,
        forward_expansion=2,
        attention_layer_num=2,
    ):
        super(FeatureFusionLayer, self).__init__()
        self.input_dim = (1 + state_dim + env_dim) * hidden_dim
        
        # Project input to hidden_dim for attention
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout, forward_expansion)
            for _ in range(attention_layer_num)
        ])
        

    def forward(self, latent_vec_batch: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        latent_vec_batch: [batch_len, max_seq_len, (1+state_dim+env_dim)*HIDDEN_DIM]
        padding_mask: [batch_len, max_seq_len], True for padding positions
        return: logits [batch_len, max_seq_len-1]
        """
        # Project to hidden_dim: [batch, seq, hidden_dim]
        x = self.input_proj(latent_vec_batch)
        
        # Apply self-attention layers
        # for attn_layer in self.self_attention_layers:
        #     x, _ = attn_layer(x, x, x, key_padding_mask=padding_mask)
        
        # Output projection: [batch, seq, hidden_dim]
        features = x
        
        # Concatenate adjacent positions for bond prediction
        # left: [batch, seq-1, hidden_dim], right: [batch, seq-1, hidden_dim]
        # left_features = features[:, :-1, :]
        # right_features = features[:, 1:, :]
        # bond_features: [batch, seq-1, hidden_dim * 2]
        # bond_features = torch.cat([left_features, right_features], dim=-1)
        
        return features

class Predictor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super(Predictor, self).__init__()
        # MLP for predicting bond cleavage (combines adjacent positions)
        self.bond_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, bond_features: torch.Tensor) -> torch.Tensor:
        # bond_features: [batch, seq, hidden_dim]
        logits = self.bond_mlp(bond_features).squeeze(-1)  # [batch, seq]
        return logits[:,:-1]  # Return logits for bonds between residues

class Model(nn.Module):
    def __init__(self, config: dict):

        super(Model, self).__init__()
        aa_type_count = len(config["seq"]["alphabet"])
        pad_index = str(config["seq"]["alphabet"]).index(str(config["seq"]["pad_char"]))
        env_dim = len(config["csv"]["env_var_col_name"])

        state_dim = len(config["csv"]["state_var_col_name"])
        self.state_norm = nn.BatchNorm1d(state_dim)
        self.env_norm = nn.BatchNorm1d(env_dim)
        self.encoder = Encoder(
            aa_type_count=aa_type_count,
            pad_index=pad_index,
            state_dim=state_dim,
            env_dim=env_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            forward_expansion=config["model"]["forward_expansion"],
            attention_layer_num=config["model"]["attention_layer_num"],
        )
        self.feature_fusion_layer = FeatureFusionLayer(
            state_dim=state_dim,
            env_dim=env_dim,
            output_dim=(int(config["seq"]["max_len"]) - 1),
            hidden_dim=config["model"]["hidden_dim"],
            dropout=config["model"]["dropout"],
            num_heads=config["model"]["num_heads"],
            forward_expansion=config["model"]["forward_expansion"],
            attention_layer_num=config["model"].get("decoder_attention_layer_num", 2),
        )
        self.predictor = Predictor(
            hidden_dim=config["model"]["hidden_dim"],
            dropout=config["model"]["dropout"])
        
        self.param_dict: dict = config
        self.param_dict["model"]["aa_type_count"] = aa_type_count
        self.param_dict["model"]["env_dim"] = env_dim

        self.param_dict["model"]["env_dim"] = state_dim

    def forward(
        self,
        seq_index_batch: torch.Tensor,
        seq_padding_mask_batch: torch.Tensor,
        state_vec_batch: torch.Tensor,
        env_vec_batch: torch.Tensor,
    ) -> torch.Tensor:
        # seq_index_batch : [batch_len,max_seq_len]
        # seq_padding_mask_batch : [batch_len,max_seq_len]
        # state_vec_batch: [batch_len,state_dim]
        # env_vec_batch: [batch_len,ENV_DIM]
        state_vec_batch = self.state_norm.forward(state_vec_batch)

        env_vec_batch = self.env_norm.forward(env_vec_batch)
        # latent_vec_batch : [batch_len,HIDDEN_DIM]
        # attn_weight_dict : dict
        latent_vec_batch = self.encoder.forward(
            seq_index_batch=seq_index_batch,
            seq_padding_mask_batch=seq_padding_mask_batch,
            state_vec_batch=state_vec_batch,
            env_vec_batch=env_vec_batch,
        )
        # bond_features : [batch_len,max_seq_len-1,2*HIDDEN_DIM]
        bond_features = self.feature_fusion_layer.forward(latent_vec_batch, padding_mask=seq_padding_mask_batch)
        # out : [batch_len,max_seq_len-1]
        out = self.predictor.forward(bond_features)
        out = out.masked_fill(seq_padding_mask_batch[:, 1:], -1e9)
        # out = out.masked_fill(seq_padding_mask_batch[:,1:],float("-inf"))
        # out : [batch,2]
        return out

    def __str__(self) -> str:
        param_dict_yaml = yaml.dump(self.param_dict, sort_keys=False)
        model_arch = super().__str__()
        return param_dict_yaml + "\n" + model_arch
