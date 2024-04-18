"""
    Source: https://github.com/thuml/Time-Series-Library/tree/main
"""
import torch
import torch.nn as nn
from torch import Tensor

from layers.transformer_layers import Encoder, EncoderLayer
from layers.attention_layers import FullAttention, AttentionLayer
from layers.embedding_layers import DataEmbedding

from models.base_imp import BaseImputer


class TransformerImputer(BaseImputer):
    """
    A transformer-based imputer model for time-series data imputation.

    This class implements a transformer architecture specifically adapted for the task of imputing
    missing values in time-series data. It includes an embedding layer for the input data, followed
    by a configurable number of encoder layers, and a final projection layer to produce the imputed
    output.

    Paper link: https://arxiv.org/abs/1706.03762

    Attributes:
        output_attention (bool): Flag to indicate whether to output attention weights.
        encoder_input (int): The number of features in the encoder input.
        model_dimension (int): The dimension of the model, specifically the size of the embeddings and hidden layers.
        embed (str): The type of embedding used.
        frequency (str): The frequency of the embeddings.
        dropout (float): Dropout rate for regularization.
        factor (int): Factor for controlling the size of the full attention mechanism.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward network.
        encoder_layers (int): The number of layers in the encoder.
        output_size (int): The number of output channels.

    Args:
        config: A configuration object containing model parameters and settings.
    """

    def __init__(self, config):
        super(TransformerImputer, self).__init__(config)
        self.output_attention = config.output_attention
        self.encoder_input = config.encoder_input
        self.model_dimension = config.model_dimension
        self.embed = config.embed
        self.frequency = config.frequency
        self.dropout = config.dropout
        self.factor = config.factor
        self.output_attention = config.output_attention
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.encoder_layers = config.encoder_layers
        self.output_size = config.output_size
        self.activation = config.activation

        # Embedding
        self.enc_embedding = DataEmbedding(c_in=self.encoder_input,
                                           d_model=self.model_dimension,
                                           embed_type=self.embed,
                                           freq=self.frequency,
                                           dropout=self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=AttentionLayer(
                        FullAttention(mask_flag=False,
                                      factor=self.factor,
                                      attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        d_model=self.model_dimension,
                        n_heads=self.n_heads),
                    d_model=self.model_dimension,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.model_dimension)
        )

        # Decoder
        self.projection = nn.Linear(in_features=self.model_dimension, out_features=self.output_size, bias=True)

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor):
        """
           Implements the imputation logic using the Transformer architecture.

           This method processes the input data through an embedding layer, followed by the encoder layers
           and a final projection layer to produce the imputed time-series data.

           Args:
               x_encoded (Tensor): The encoded input tensor representing time-series data.
               x_mark_encoded (Tensor): Encoded auxiliary time-related information.
               mask (Tensor): A mask tensor indicating the presence of missing values.

           Returns:
               Tensor: The output tensor after imputation.
        """
        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out
