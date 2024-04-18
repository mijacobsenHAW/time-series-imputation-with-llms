from pydantic import BaseModel


class NetworkConfig(BaseModel):
    """
        Args:
            seq_len: sequence length
            patch_size: size of the patch

    """
    seq_len: int
    patch_size: int

