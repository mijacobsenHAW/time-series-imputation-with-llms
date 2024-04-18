import torch
import torch.nn as nn
from torch import Tensor
from models.base_imp import BaseImputer
from layers.embedding_layers import DataEmbedding


class LlmImputer(BaseImputer):
    """
    lmImputer is an abstract base class for imputers based on large language models (LLMs),
    providing a common framework for different types of LLM-based imputation strategies.

    This class sets up the essential components shared across various LLM-based imputers,
    such as the embedding layers and output projection. It is designed to be subclassed
    by specific imputer implementations that utilize different LLM architectures for
    time-series data imputation.

    Attributes:
        model_dimension (int): The dimension of the model, usually matching the size of embeddings and hidden layers.
        ln_proj (nn.LayerNorm): Layer normalization applied before the output layer.
        encoder_inp (int): The size of the encoder input, representing the number of attributes/features.
        output_size (int): The size of the output layer, typically corresponding to the number of features to be imputed.
        patch_size (int): The size of patches into which the input is divided, applicable in certain LLM models.
        embed (str): The type of embedding used.
        freq (str): The frequency of the embeddings.
        dropout (float): Dropout rate for regularization.
        mlp (int): Indicates the presence and configuration of a multilayer perceptron layer.
        out_layer (nn.Linear): Linear output layer for the imputation model.
        enc_embedding (DataEmbedding): Embedding layer for input data processing.

    Args:
        config: A configuration object containing parameters and settings for the imputer model.
    """

    def __init__(self, config):
        super(LlmImputer, self).__init__(config)
        self.model_dimension = config.model_dimension
        self.ln_proj = nn.LayerNorm(self.model_dimension)
        self.encoder_inp = config.encoder_input
        self.output_size = config.output_size
        self.patch_size = config.patch_size
        self.embed = config.embed
        self.freq = config.frequency
        self.dropout = config.dropout
        self.mlp = config.mlp

        self.out_layer = nn.Linear(self.model_dimension, self.output_size, bias=True)
        self.enc_embedding = DataEmbedding(c_in=(self.encoder_inp * self.patch_size),
                                           d_model=self.model_dimension, embed_type=self.embed,
                                           freq=self.freq, dropout=self.dropout)

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Abstract method for imputation to be implemented by subclasses.

        This method defines the interface for the imputation process using LLM-based models.
        Subclasses should implement the specific logic for handling the imputation based on
        their respective LLM architectures.

        Args:
            x_encoded (Tensor): The encoded input tensor representing the time-series data.
            x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
            mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

        Returns:
            torch.Tensor: The output tensor after imputation by the subclass implementation.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Subclass must implement this method")
