model=GptImputer
run_name="components_tuning_3_layers_05_tmp"

# Masking rate 50%, no fine-tuning
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/electricity/ \
#  --data_path electricity.csv \
#  --data custom \
#  --mask_rate 0.5 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 321 \
#  --decoder_input 321 \
#  --output_size 321 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Electricity_05_1703' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "no_ft"

# Masking rate 50%, attention
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_05_1703_attention' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention"

# Masking rate 50%, ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_05_1703_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "ffn"

#
## Masking rate 50%, add_layernorm
#python -u run.py \
#--run_name $run_name \
#--is_training True \
#--model $model \
#--root ./data/electricity/ \
#--data_path electricity.csv \
#--data custom \
#--mask_rate 0.5 \
#--sequence_len 96 \
#--label_len 0 \
#--prediction_len 0 \
#--encoder_input 321 \
#--decoder_input 321 \
#--output_size 321 \
#--gpt_layers 3 \
#--batch_size 16 \
#--model_dimension 768 \
#--description 'Experiment_Electricity_05_1703_add_layernorm' \
#--iterations 1 \
#--mlp 1 \
#--learning_rate 0.001 \
#--lora True \
#--components "add_layernorm"
#
## Masking rate 50%, attention_add_layernorm
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/electricity/ \
#  --data_path electricity.csv \
#  --data custom \
#  --mask_rate 0.5 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 321 \
#  --decoder_input 321 \
#  --output_size 321 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Electricity_05_1703_attention_add_layernorm' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "attention_add_layernorm"

# Masking rate 50%, attention_ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_05_1703_attention_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_ffn"

# Masking rate 50%, ffn_add_layernorm
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/electricity/ \
#  --data_path electricity.csv \
#  --data custom \
#  --mask_rate 0.5 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 321 \
#  --decoder_input 321 \
#  --output_size 321 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Electricity_05_1703_ffn_add_layernorm' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "ffn_add_layernorm"
#
## Masking rate 50%, all
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/electricity/ \
#  --data_path electricity.csv \
#  --data custom \
#  --mask_rate 0.5 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 321 \
#  --decoder_input 321 \
#  --output_size 321 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Electricity_05_1703_attention_add_layernorm_ffn' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "attention_add_layernorm_ffn"

