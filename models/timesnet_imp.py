"""
    TimesNet Implementation
    Title of Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
    Link to Paper: https://arxiv.org/abs/2210.02186
    Source of Code: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
"""
import torch.nn as nn
from torch import Tensor
from models.base_imp import BaseImputer
from layers.times_block import TimesBlock
from layers.embedding_layers import DataEmbedding


class TimesNetImputer(BaseImputer):
    """
    TimesNetImputer is a neural network model designed for imputing missing values in time-series data.
    It extends the BaseImputer class and leverages the TimesBlock architecture for processing the sequences.

    This class includes an embedding layer for data processing, a sequence of TimesBlock layers for
    handling time-series data, followed by normalization and denormalization steps, and a final projection
    layer for output. It is designed to handle various lengths and patterns of missing data in time-series sequences.

    Attributes:
        model (nn.ModuleList): A list of TimesBlock layers for time-series processing.
        enc_embedding (DataEmbedding): Embedding layer for input data.
        layer (int): The number of TimesBlock layers used.
        layer_norm (nn.LayerNorm): Layer normalization.
        projection (nn.Linear): Linear projection layer for output.

    Args:
        config: A configuration object with model parameters like sequence length, prediction length, and layer specifications.
    """
    def __init__(self, config):
        super(TimesNetImputer, self).__init__(config)
        # self.seq_len = config.seq_len
        # self.label_len = config.label_len
        # self.pred_len = config.pred_len
        self.encoder_layers = config.encoder_layers
        self.encoder_input = config.encoder_input
        self.model_dimension = config.model_dimension
        self.embed = config.embed
        self.frequency = config.frequency
        self.dropout = config.dropout
        self.output_size = config.output_size
        self.top_k = config.top_k
        self.d_ff = config.d_ff
        self.num_kernels = config.num_kernels

        self.model = nn.ModuleList([TimesBlock(sequence_len=self.sequence_len,
                                               prediction_len=self.prediction_len,
                                               top_k=self.top_k,
                                               model_dimension=self.model_dimension,
                                               d_ff=self.d_ff,
                                               num_kernels=self.num_kernels,
                                               device=config.apple_device)
                                    for _ in range(self.encoder_layers)])

        self.enc_embedding = DataEmbedding(c_in=self.encoder_input,
                                           d_model=self.model_dimension,
                                           embed_type=self.embed,
                                           freq=self.frequency,
                                           dropout=self.dropout)

        self.layer_norm = nn.LayerNorm(self.model_dimension)

        self.projection = nn.Linear(self.model_dimension, self.output_size, bias=True)

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> Tensor:
        """
        Implements the imputation process using the TimesNet architecture.

        The method involves normalization of the input data, processing through the TimesNet layers,
        and then denormalization of the output. The imputation is specifically tailored for time-series data.

        Args:
            x_encoded (Tensor): The encoded input tensor.
            x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
            mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

        Returns:
            Tensor: The output tensor after imputation, representing the imputed time-series data.
        """

        x_encoded, stdev, means = self.normalization_non_stationary_transformer(x_encoded=x_encoded, mask=mask)

        # embedding
        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)  # [B,T,C]

        # TimesNet
        for i in range(self.encoder_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        decoded_out = self.projection(enc_out)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)
        return decoded_out
