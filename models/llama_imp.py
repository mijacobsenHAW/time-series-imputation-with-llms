import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers.models.llama.modeling_llama import LlamaModel
from peft import get_peft_model, LoraConfig


class LlamaImputer(LlmImputer):
    """
    LlamaImputer is a class extending LlmImputer, designed to use the Llama model for the purpose of time-series data imputation.

    This class integrates the LlamaModel, a transformer-based model, for imputing missing values in time-series data.
    It includes specific configurations for the Llama model and applies PEFT (Parameter-efficient Fine-tuning) modifications
    to adapt it for handling the imputation task effectively in large-scale time-series datasets.

    Attributes:
        llama_layers (int): The number of layers in the Llama model.
        llama2 (LlamaModel): The Llama model loaded with specific configurations and weights.
        peft_config (LoraConfig): Configuration for the PEFT modifications applied to the Llama model.

    Args:
        config: A configuration object containing parameters and settings for the imputer model.
    """
    def __init__(self, config):
        super(LlamaImputer, self).__init__(config)
        self.llama_layers = config.llama_layers
        self.llama2 = LlamaModel.from_pretrained(
            pretrained_model_name_or_path='./model_weights/Llama-2-7b-chat-hf',
            output_attentions=True,
            output_hidden_states=True,
            local_files_only=True,
            # load_in_4bit=True
        )

        self.llama2.layers = self.llama2.layers[:self.llama_layers]
        self.lora = config.lora

        if self.lora:
            print("Lora")
            self.peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none"
            )
        self.llama2 = get_peft_model(model=self.llama2, peft_config=self.peft_config)

        print("Before param.requires_grad setting")
        self.llama2.print_trainable_parameters()
        for i, (name, param) in enumerate(self.llama2.named_parameters()):
            if 'input_layernorm' in name or 'post_attention_layernorm' in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")
        self.trainable_params, self.all_param = self.llama2.get_nb_trainable_parameters()
        self.llama2.print_trainable_parameters()

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the Llama model.

        The method includes normalization of the input data, processing it through the Llama model with PEFT modifications,
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
        outputs = self.llama2(inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)

        return decoded_out
