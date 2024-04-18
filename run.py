# This file is used to run the imputation experiment for the specified model.
import torch.backends.mps
import argparse
from experiments.imputation import Imputation

parser = argparse.ArgumentParser(description='Run imputation experiment for specified model')

parser.add_argument('--run_name', type=str, default='default', help='name for run (e.g. date), is mandatory for organizing the result folder')
parser.add_argument('--description', type=str, default='desc', help='Description for naming')
parser.add_argument('--is_training', type=str, choices=['True', 'False'], help='Flag to indicate if the model is in training mode.')
parser.add_argument('--model', type=str, default='model_name', help='Name of the model to use.')
parser.add_argument('--data', type=str, default='Etth1', help='Type of data used.')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Path to the data.')
parser.add_argument('--root', type=str, default='./data/ETT/', help='Root directory path.')
parser.add_argument('--frequency', type=str, default='h', help='Frequency for data embedding.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size for data embedding.')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Path to save checkpoints.')
parser.add_argument('--sequence_len', type=int, default=96, help='Length of the input sequences.')
parser.add_argument('--label_len', type=int, default=0, help='Length of the labels.')
parser.add_argument('--prediction_len', type=int, default=0, help='Length of the prediction sequence.')
parser.add_argument('--mask_rate', type=float, default=0.1, help='Rate of masking for imputation.')
parser.add_argument('--iterations', type=int, default=3, help='Number of training iterations.')
parser.add_argument('--model_id', type=str, default='model_data_maskrate', help='Unique identifier for the model.')
parser.add_argument('--encoder_input', type=int, default=7, help='Input size for the encoder.')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers in the encoder.')
parser.add_argument('--decoder_input', type=int, default=7, help='Input size for the decoder.')
parser.add_argument('--decoder_layers', type=int, default=1, help='Number of layers in the decoder.')
parser.add_argument('--output_size', type=int, default=1, help='Output size of the model.')
parser.add_argument('--model_dimension', type=int, default=512, help='Dimension of the model embeddings.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the model.')
parser.add_argument('--mlp', type=int, default=0, help='Use MLP layers.')
parser.add_argument('--percent', type=int, default=100, help='Percentage for amount of training data.')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Size of the training batch.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use.')
parser.add_argument('--adjust_lr', type=str, default='type1', help='Method for adjusting learning rate.')
parser.add_argument('--use_gpu', type=bool, default=True, help='Flag to use GPU for training.')
parser.add_argument('--embed', type=str, default='timeF', help='Type of embedding to use.')
parser.add_argument('--lora', type=bool, default=True, help='Train config with lora or not')
parser.add_argument('--peft_type', type=str, default='lora', help='Type of peft method to use.')
parser.add_argument('--components', type=str, default='default', help='Components to be trained')

# Model specific configurations
parser.add_argument('--output_attention', type=bool, default=False, help='Flag to output attention weights.')
parser.add_argument('--factor', type=int, default=5, help='Factor for something in Transformer Imputer.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed-forward layers.')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function to use.')

# Saits Imputer
parser.add_argument('--k', type=int, default=64, help='Dimension of key vectors in self-attention.')
parser.add_argument('--v', type=int, default=64, help='Dimension of value vectors in self-attention.')
parser.add_argument('--mit_weight', type=int, default=1, help='Weight for MIT in Saits Imputer.')
parser.add_argument('--ort_weight', type=int, default=1, help='Weight for ORT in Saits Imputer.')
parser.add_argument('--diagonal_attention_mask', type=bool, default=False,
                    help='Use diagonal attention mask in Saits Imputer.')

# TimesNet Imputer
parser.add_argument('--top_k', type=int, default=5, help='Top K for TimesNet Imputer.')
parser.add_argument('--num_kernels', type=int, default=10, help='Number of kernels in TimesNet Imputer.')
parser.add_argument('--apple_device', type=bool, default=False, help='Use Apple device in TimesNet Imputer.')

# Large Models Imputer Parameters
parser.add_argument('--gpt_layers', type=int, default=12, help='Number of GPT layers.')
parser.add_argument('--llama_layers', type=int, default=32, help='Number of layers in Llama Imputer.')
parser.add_argument('--bert_layers', type=int, default=12, help='Number of layers to be used of BERT.')
parser.add_argument('--phi_layers', type=int, default=32, help='Number of layers to be used of Phi.')
parser.add_argument('--gemma_layers', type=int, default=28, help='Number of layers to be used of Gemma.')
parser.add_argument('--t5_layers', type=int, default=12, help='Number of layers to be used of T5.')
parser.add_argument('--bart_layers', type=int, default=6, help='Number of layers to be used of BART.')

args = parser.parse_args()

if __name__ == '__main__':
    args.model_id = args.model + '_' + args.data + '_' + str(args.mask_rate)
    print(f'#### Run Experiment {args.model_id}')

    args.use_gpu = True if (torch.cuda.is_available() or torch.backends.mps.is_available()) and args.use_gpu else False

    print('Args in experiment: ' + "\n" + str(args))
    is_training = args.is_training == 'True'
    if is_training:
        for i in range(args.iterations):
            # setting record of experiments
            setting = '{}_md{}_{}_iteration_{}'.format(
                args.model_id,
                args.model_dimension,
                args.description,
                i
            )

            imputation_exp = Imputation(args,
                                        diagonal_attention_mask=args.diagonal_attention_mask)

            print(">>>> Start training: {} <<<<".format(setting))

            imputation_exp.train(setting)

            print('>>>> Testing : {} <<<<'.format(setting))
            imputation_exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_md{}_{}_iteration_{}'.format(
            args.model_id,
            args.model_dimension,
            args.description,
            ii
        )

        imputation_exp = Imputation(args,
                                    diagonal_attention_mask=args.diagonal_attention_mask)
        print('>>>> Testing : {} <<<<'.format(setting))
        imputation_exp.test(setting, test=1)
        torch.cuda.empty_cache()
