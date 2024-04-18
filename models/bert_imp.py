import torch
from torch import Tensor
from models.llm_imp import LlmImputer
from transformers import BertModel
from peft import get_peft_model, LoraConfig


class BertImputer(LlmImputer):
    """
    BertImputer is a class that extends LlmImputer, utilizing the BERT model for imputing missing values in time-series data.

    This class adapts the BertModel for the specific task of imputing missing values, leveraging BERT's powerful natural language processing capabilities. It is designed to handle the complexities and nuances in time-series data by employing a pre-trained BERT model, with the flexibility to specify the number of BERT layers to use.
    The model is also equipped with PEFT (Parameter-efficient Fine-tuning) modifications for enhanced performance and efficiency in handling large-scale time-series datasets.

    Attributes:
        bert_layers (int): The number of BERT layers to be used for the imputation task.
        bert (BertModel): The BERT model loaded with specific configurations for imputation.
        peft_config (LoraConfig): Configuration for the PEFT modifications applied to the BERT model.
        lora (bool): A flag indicating whether to use the LORA mechanism for the BERT model.
        trainable_params (int): The number of trainable parameters in the BERT model.
        all_param (int): The total number of parameters in the BERT model.
    Args:
        config: A configuration object containing model parameters and settings, including the number of BERT layers, GPU usage, and other relevant information.
    """

    def __init__(self, config):
        super(BertImputer, self).__init__(config)
        self.bert_layers = config.bert_layers
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased",
                                              output_attentions=True,
                                              output_hidden_states=True)

        self.bert.encoder.layer = self.bert.encoder.layer[:self.bert_layers]
        self.lora = config.lora

        if self.lora:
            print("Lora")
            self.peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["key", "query", "value", "attention.output.dense"],
                lora_dropout=0.05,
                bias="none"
            )

        self.bert = get_peft_model(model=self.bert, peft_config=self.peft_config)

        print("Before param.requires_grad setting")
        self.bert.print_trainable_parameters()
        for i, (name, param) in enumerate(self.bert.named_parameters()):
            if "position_embeddings" in name:
                param.requires_grad = True
            elif "LayerNorm" in name:
                param.requires_grad = True
            elif 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Name der Schicht {name} \n Schicht trainierbar: {param.requires_grad}")

        self.trainable_params, self.all_param = self.bert.get_nb_trainable_parameters()
        self.bert.print_trainable_parameters()

        if config.use_gpu:
            if torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                self.bert.to(device=mps_device)
            else:
                print("MPS device not found.")

    def imputation(self, x_encoded: Tensor, x_mark_encoded: Tensor, mask: Tensor) -> torch.Tensor:
        """
        Implements the imputation process using the BERT model.

        The method involves normalization of the input data, processing it through the BERT model, and then denormalization to produce the imputed time-series data. It is specifically designed to handle complex patterns of missing data in time-series sequences.

        Args:
            x_encoded (Tensor): The encoded input tensor representing the time-series data.
            x_mark_encoded (Tensor): Encoded auxiliary markers or time-related information.
            mask (Tensor): A mask tensor indicating the presence of missing values in the input data.

        Returns:
            torch.Tensor: The output tensor after imputation, representing the imputed time-series data.
        """
        x_encoded, stdev, means = self.normalization_non_stationary_transformer(x_encoded, mask)

        enc_out = self.enc_embedding(x_encoded, x_mark_encoded)
        outputs = self.bert(inputs_embeds=enc_out).last_hidden_state
        outputs = self.ln_proj(outputs)
        decoded_out = self.out_layer(outputs)

        decoded_out = self.denormalization_non_stationary_transformer(decoded_out=decoded_out, stdev=stdev, means=means)
        return decoded_out
