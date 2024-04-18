import torch
import torch.nn as nn
from torch import Tensor


class BaseImputer(nn.Module):
    """
    Base class for implementing imputation models in PyTorch.
    This class is designed to be subclassed for specific imputation strategies.

    Attributes:
        config: A configuration object containing parameters for the imputer.
        sequence_len: The length of input sequences.
        label_len: The length of labels in the sequences.

    Args:
        config: A configuration object with attributes like sequence_len and label_len.
    """
    def __init__(self, config):
        super(BaseImputer, self).__init__()
        self.config = config
        self.sequence_len = config.sequence_len
        self.label_len = config.label_len
        self.prediction_len = config.prediction_len

    def forward(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> Tensor:
        """
        Performs a forward pass through the imputation model.

        Args:
            x_encoded (Tensor): The encoded input tensor.
            x_mark_encoded (Tensor): Encoded auxiliary information (e.g., time marks).
            mask (Tensor): A mask tensor indicating the presence of missing values.

        Returns:
            Tensor: The output tensor after imputation.
        """
        dec_out = self.imputation(x_encoded, x_mark_encoded, mask)
        return dec_out

    def normalization_non_stationary_transformer(self, x_encoded, mask):
        """
        Applies normalization to the input tensor using a non-stationary transformer approach.

        Args:
            x_encoded (Tensor): The encoded input tensor to be normalized.
            mask (Tensor): A mask tensor indicating the presence of missing values.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the normalized tensor,
            the standard deviation, and the mean used for normalization.
        """

        means = torch.sum(x_encoded, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_encoded = x_encoded - means
        x_encoded = x_encoded.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_encoded * x_encoded, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_encoded /= stdev

        return x_encoded, stdev, means

    def denormalization_non_stationary_transformer(self, decoded_out, stdev, means):
        """
        Reverses the normalization applied to the tensor.

        Args:
            decoded_out (Tensor): The tensor to be denormalized.
            stdev (Tensor): The standard deviation used in the normalization step.
            means (Tensor): The mean used in the normalization step.

        Returns:
            Tensor: The denormalized tensor.
        """

        decoded_out = decoded_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.sequence_len, 1))
        decoded_out = decoded_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.sequence_len, 1))

        return decoded_out

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError("Subclass must implement this method")
