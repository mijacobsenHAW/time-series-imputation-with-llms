import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers.models.phi.modeling_phi import PhiModel
from peft import get_peft_model, LoraConfig


class PhiImputer(LlmImputer):
    """
    PhiImputer is a specialized imputer class that extends LlmImputer, integrating the Phi model for time-series data imputation.

    This class adapts the Phi model, a transformer-based model, for the task of imputing missing values in time-series data.
    It incorporates the PhiModel with PEFT (Parameter-efficient Fine-tuning) modifications for enhanced performance
    and efficiency in handling large-scale time-series datasets.

    Attributes:
        phi_layers (int): The number of layers in the Phi model.
        phi (PhiModel): The Phi model loaded with specific configurations and weights.
        peft_config (LoraConfig): Configuration for the PEFT modifications applied to the Phi model.

    Args:
        config: A configuration object containing parameters and settings for the imputer model.
    """

    def __init__(self, config):
        super(PhiImputer, self).__init__(config)
        self.phi_layers = config.phi_layers
        self.phi = PhiModel.from_pretrained(
            pretrained_model_name_or_path='./model_weights/phi-2',
            output_attentions=True,
            output_hidden_states=True,
            local_files_only=True,
            # load_in_4bit=True
        )

        self.phi.layers = self.phi.layers[:self.phi_layers]
        self.lora = config.lora

        if self.lora:
            print("Lora")
            self.peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["k_proj", "q_proj", "v_proj", "dense"],
                lora_dropout=0.05,
                bias="none"
            )
        self.phi = get_peft_model(model=self.phi, peft_config=self.peft_config)
        self.phi.print_trainable_parameters()

        for i, (name, param) in enumerate(self.phi.named_parameters()):
            if 'input_layernorm' in name:
                param.requires_grad = True
            elif 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")
        self.trainable_params, self.all_param = self.phi.get_nb_trainable_parameters()
        self.phi.print_trainable_parameters()

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the Phi model.

        The method includes normalization of the input data, processing it through the Phi model with PEFT modifications,
        and then denormalization to produce the imputed time-series data. It is specifically designed to handle complex
        patterns of missing data in time-series sequences.

       Args:
           x_encoded (Tensor): The encoded input tensor.
           x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
           mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

       Returns:
           Tensor: The output tensor after imputation, representing the imputed time-series data.
       """
        x_encoded, stdev, means = self.normalization_non_stationary_transformer(x_encoded, mask)

        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)
        outputs = self.phi(inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)

        return decoded_out
