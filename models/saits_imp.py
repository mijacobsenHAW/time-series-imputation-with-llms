"""
    SAITS Implementation
    Title of Paper: SAITS: Self-attention-based imputation for time series
    Link to Paper: https://www.sciencedirect.com/science/article/abs/pii/S0957417423001203?via%3Dihub
    Source of Code: https://github.com/WenjieDu/SAITS/tree/main
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers.saits_layers import EncoderLayer, PositionalEncoding
from utils.saits_utils import masked_mae_cal


class SaitsImputer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.encoder_layers = config.encoder_layers  # before n_groups, n_group_inner_layers
        self.sequence_len = config.sequence_len  # before d_time
        self.encoder_input = config.encoder_input  # before d_feature, actual_d_feature, number of features
        self.actual_d_feature = self.encoder_input * 2
        self.model_dimension = config.model_dimension  # before d_model
        self.d_ff = config.d_ff  # before d_inner
        self.n_heads = config.n_heads  # before n_head
        self.k = config.k
        self.v = config.v
        self.dropout = nn.Dropout(p=config.dropout)

        self.mit_weight = config.mit_weight  # default float = 1 before via kwargs
        self.ort_weight = config.ort_weight  # default float = 1 aus Pypots
        # self.device = kwargs["device"]
        # self.param_sharing_strategy = kwargs["param_sharing_strategy"]

        self.device = "cpu"
        if config.use_gpu:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    device=self.device,
                    d_time=self.sequence_len,
                    d_feature=self.actual_d_feature,
                    d_model=self.model_dimension,
                    d_inner=self.d_ff,
                    n_head=self.n_heads,
                    d_k=self.k,
                    d_v=self.v,
                    dropout=self.dropout,
                    attn_dropout=0,
                    **kwargs
                )
                for _ in range(self.encoder_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    device=self.device,
                    d_time=self.sequence_len,
                    d_feature=self.actual_d_feature,
                    d_model=self.model_dimension,
                    d_inner=self.d_ff,
                    n_head=self.n_heads,
                    d_k=self.k,
                    d_v=self.v,
                    dropout=self.dropout,
                    attn_dropout=0,
                    **kwargs
                )
                for _ in range(self.encoder_layers)
            ]
        )

        self.position_enc = PositionalEncoding(self.model_dimension, n_position=self.sequence_len)

        # for the 1st block
        self.embedding_1 = nn.Linear(self.actual_d_feature, self.model_dimension)
        self.reduce_dim_z = nn.Linear(self.model_dimension, self.encoder_input)

        # for the 2nd block
        self.embedding_2 = nn.Linear(self.actual_d_feature, self.model_dimension)
        self.reduce_dim_beta = nn.Linear(self.model_dimension, self.encoder_input)
        self.reduce_dim_gamma = nn.Linear(self.encoder_input, self.encoder_input)

        # for the 3rd block
        self.weight_combine = nn.Linear(in_features=self.encoder_input + self.sequence_len,
                                        out_features=self.encoder_input)

    def imputation(self, x_encoded: Tensor, mask: Tensor):
        # 1. DMSA block
        x1 = torch.cat(tensors=[x_encoded, mask], dim=2)
        x1 = self.embedding_1(x1)

        enc_out = self.dropout(self.position_enc(x1))

        for encoder_layer in self.layer_stack_for_first_block:
            for _ in range(self.encoder_layers):
                enc_out, _ = encoder_layer(enc_out)

        x1_tilde = self.reduce_dim_z(enc_out)
        x = mask * x_encoded + (1 - mask) * x1_tilde

        # 2. DMSA block
        x2 = torch.cat(tensors=[x, mask], dim=2)
        x2 = self.embedding_2(x2)

        enc_out = self.position_enc(x2)
        attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            for _ in range(self.encoder_layers):
                enc_out, attn_weights = encoder_layer(enc_out)

        x2_tilde = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_out)))

        attn_weights = attn_weights.squeeze(dim=1)

        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat(tensors=[mask, attn_weights], dim=2))
        )

        x3_tilde = (1 - combining_weights) * x2_tilde + combining_weights * x1_tilde

        dec_out = mask * x_encoded + (1 - mask) * x3_tilde
        return dec_out, [x1_tilde, x2_tilde, x3_tilde]

    def forward(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor, stage: str = 'train'):
        # X, masks = inputs["X"], inputs["missing_mask"]
        x = x_encoded
        masks = mask
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.imputation(x_encoded=x, mask=masks)

        reconstruction_loss += masked_mae_cal(X_tilde_1, x, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, x, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, x, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                X_tilde_3, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        # return imputed_data
        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
