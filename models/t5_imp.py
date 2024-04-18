import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers import T5Model
from peft import get_peft_model, LoraConfig


class T5Imputer(LlmImputer):
    """
    T5Imputer is a class that extends LlmImputer, utilizing the T5 model for imputing missing values in time-series data.

    This class adapts the T5 model for the specific task of imputing missing values, leveraging T5's powerful natural language processing capabilities. It is designed to handle the complexities and nuances in time-series data by employing a pre-trained T5 model, with the flexibility to specify the number of T5 layers to use.
    The model is also equipped with PEFT (Parameter-efficient Fine-tuning) modifications for enhanced performance and efficiency in handling large-scale time-series datasets.

    Attributes:
        t5_layers (int): The number of T5 layers to be used for the imputation task.
        t5 (T5Model): The T5 model loaded with specific configurations for imputation.
        peft_config (LoraConfig): Configuration for the PEFT modifications applied to the T5 model.
        lora (bool): A flag indicating whether to use the LORA mechanism for the T5 model.
        trainable_params (int): The number of trainable parameters in the T5 model.
        all_param (int): The total number of parameters in the T5 model.
    Args:
        config: A configuration object containing model parameters and settings, including the number of BERT layers, GPU usage, and other relevant information.
    """

    def __init__(self, config):
        super(T5Imputer, self).__init__(config)
        self.t5_layers = config.t5_layers
        self.t5 = T5Model.from_pretrained(pretrained_model_name_or_path="google-t5/t5-base",
                                          output_attentions=True,
                                          output_hidden_states=True)

        self.t5.encoder.block = self.t5.encoder.block[:self.t5_layers]
        self.t5.decoder.block = self.t5.decoder.block[:self.t5_layers]
        self.lora = config.lora

        if self.lora:
            print("Lora")
            self.peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["SelfAttention.k",
                                "SelfAttention.o",
                                "SelfAttention.q",
                                "SelfAttention.v",
                                "SelfAttention.relative_attention_bias",
                                "EncDecAttention.k",
                                "EncDecAttention.o",
                                "EncDecAttention.q",
                                "EncDecAttention.v",
                                "EncDecAttention.relative_attention_bias"],
                lora_dropout=0.05,
                bias="none"
            )
        self.t5 = get_peft_model(model=self.t5, peft_config=self.peft_config)

        print("Before param.requires_grad setting")
        self.t5.print_trainable_parameters()
        for i, (name, param) in enumerate(self.t5.named_parameters()):
            if "layer_norm" in name:
                param.requires_grad = True
            elif 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")

        self.trainable_params, self.all_param = self.t5.get_nb_trainable_parameters()
        self.t5.print_trainable_parameters()

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the T5 model.

        This method performs the imputation process using the T5 model, taking the encoded input tensor, encoded auxiliary markers, and mask tensor as input. It then returns the output tensor after imputation, representing the imputed time-series data.

        Args:
            x_encoded (Tensor): The encoded input tensor representing the time-series data.
            x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
            mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

        Returns:
            torch.Tensor: The output tensor after imputation, representing the imputed time-series data.
        """
        x_encoded, stdev, means = self.normalization_non_stationary_transformer(x_encoded, mask)

        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)
        outputs = self.t5(inputs_embeds=enc_out, decoder_inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)
        return decoded_out
