import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from peft import get_peft_model, LoraConfig, AdaLoraConfig, IA3Config
from utils.utils import count_trainable_parameters


class GptImputer(LlmImputer):
    """
    GptImputer is an imputation class that extends LlmImputer, utilizing a modified GPT-2 model for time-series data imputation.

    This class adapts the GPT2Model for the specific task of imputing missing values in time-series data. It includes configuration
    for selectively freezing and unfreezing layers of the model to fine-tune its performance for the imputation task.

    Attributes:
        gpt_layers (int): The number of GPT-2 layers to be used in the model.
        gpt2 (GPT2Model): The modified GPT-2 model instance.

    Args:
        config: A configuration object containing model parameters and settings.
    """

    def __init__(self, config):
        super(GptImputer, self).__init__(config)
        self.gpt_layers = config.gpt_layers
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model_name_or_path='gpt2',
                                              output_attentions=True,
                                              output_hidden_states=True)
        self.lora = config.lora
        self.components = config.components
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        # specific configurations for different experiments

        if self.lora and self.components == "default":
            # default experiment with lora
            if config.peft_type == "lora":
                print("Lora")
                self.peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["attn.c_attn", "attn.c_proj", "attn.bias"],
                    lora_dropout=0.05,
                    bias="none"
                )
                self.gpt2 = get_peft_model(model=self.gpt2, peft_config=self.peft_config)

                self.gpt2.print_trainable_parameters()
                for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                    if 'ln_1' in name or 'ln_2' in name or 'wpe' in name:
                        param.requires_grad = True
                    elif 'lora' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")
            elif config.peft_type == "ada_lora":
                # experiment with different peft types
                print("AdaLora")
                self.peft_config = AdaLoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["attn.c_attn", "attn.c_proj", "attn.bias"],
                    lora_dropout=0.05,
                    bias="none"
                )
            elif config.peft_type == "ia3":
                print("IA3")
                self.peft_config = IA3Config(
                    target_modules=["attn.c_attn", "attn.c_proj", "attn.bias"]
                )
            elif config.peft_type == "plain_lora":
                print("plain_lora")
                self.peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["attn.c_attn", "attn.c_proj", "attn.bias"],
                    lora_dropout=0.05,
                    bias="none"
                )
            elif config.peft_type == "qlora":
                print("QLora")
                # source: https://pytorch.org/blog/finetune-llms/
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )

                self.gpt2 = GPT2Model.from_pretrained(pretrained_model_name_or_path='gpt2',
                                                      output_attentions=True,
                                                      output_hidden_states=True,
                                                      quantization_config=quantization_config)
                self.gpt2 = prepare_model_for_kbit_training(self.gpt2)
                self.gpt2.h = self.gpt2.h[:self.gpt_layers]
                self.peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["attn.c_attn", "attn.c_proj", "attn.bias"],
                    lora_dropout=0.05,
                    bias="none"
                )

            self.gpt2 = get_peft_model(model=self.gpt2, peft_config=self.peft_config)
            self.gpt2.print_trainable_parameters()
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln_1' in name or 'ln_2' in name or 'wpe' in name:
                    param.requires_grad = True
                elif 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.trainable_params, self.all_param = self.gpt2.get_nb_trainable_parameters()
            self.gpt2.print_trainable_parameters()

        else:
            print("Component Tuning", self.components)
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                """
                    Mapping: 
                        ffn = mlp
                        attn = attn
                        add_layernorm = ln 
                """
                if 'wte' in name or 'wpe' in name:
                    param.requires_grad = True
                    print(name, param.requires_grad)
                elif 'attention' in self.components and 'attn' in name:
                    param.requires_grad = True
                    print(name, param.requires_grad)
                elif 'ffn' in self.components and 'mlp' in name:
                    param.requires_grad = True
                    print(name, param.requires_grad)
                elif 'add_layernorm' in self.components and 'ln' in name:
                    param.requires_grad = True
                    print(name, param.requires_grad)
                else:
                    param.requires_grad = False
            self.all_param, self.trainable_params = count_trainable_parameters(model=self.gpt2)
            print(f"Trainable Parameters: {self.trainable_params} All Parameters: {self.all_param}")

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the GPT-2 model.

        The method includes normalization of the input data, processing through the GPT-2 model,
        and then denormalization to produce the imputed time-series data. It is specifically designed
        to handle complex patterns of missing data in time-series sequences.

        Args:
            x_encoded (Tensor): The encoded input tensor representing the time-series data.
            x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
            mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

        Returns:
            torch.Tensor: The output tensor after imputation, representing the imputed time-series data.
        """
        x_encoded, stdev, means = self.normalization_non_stationary_transformer(x_encoded, mask)

        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)
        return decoded_out
