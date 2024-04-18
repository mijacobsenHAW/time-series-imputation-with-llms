import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers.models.gemma.modeling_gemma import GemmaModel
from peft import get_peft_model, LoraConfig


class GemmaImputer(LlmImputer):
    """
    The GemmaImputer class is a subclass of the LlmImputer class, which is used to implement the imputation process using
    the Llama model. The class includes the necessary methods to handle the imputation process, including normalization
    and denormalization of the input data, and the imputation process using the Llama model with PEFT modifications.

    Attributes:
        gemma_layers (int): The number of layers in the gemma model.
        gemma (LlamaModel): The Gemma model loaded with specific configurations and weights.
        peft_config (LoraConfig): Configuration for the PEFT modifications applied to the Llama model.

    Args:
        config (dict): A dictionary containing the configuration parameters for the imputer model.
    """
    def __init__(self, config):
        super(GemmaImputer, self).__init__(config)
        self.gemma_layers = config.gemma_layers
        self.gemma = GemmaModel.from_pretrained(
            pretrained_model_name_or_path='./model_weights/gemma-7b-it',
            output_attentions=True,
            output_hidden_states=True,
            local_files_only=True,
            # load_in_4bit=True
        )

        self.gemma.layers = self.gemma.layers[:self.gemma_layers]
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
        self.gemma = get_peft_model(model=self.gemma, peft_config=self.peft_config)

        print("Before param.requires_grad setting")
        self.gemma.print_trainable_parameters()
        for i, (name, param) in enumerate(self.gemma.named_parameters()):
            if 'input_layernorm' in name or 'post_attention_layernorm' in name:
                param.requires_grad = True
            elif 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")
        self.trainable_params, self.all_param = self.gemma.get_nb_trainable_parameters()
        self.gemma.print_trainable_parameters()

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the Gemma model.

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
        outputs = self.gemma(inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)

        return decoded_out
