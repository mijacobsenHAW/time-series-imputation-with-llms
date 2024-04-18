model=GptImputer
run_name="components_tuning_3_layers_0375"

# Masking rate 37.5%, no fine-tuning
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/electricity/ \
#  --data_path electricity.csv \
#  --data custom \
#  --mask_rate 0.375 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 321 \
#  --decoder_input 321 \
#  --output_size 321 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Electricity_0375_1703' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "no_ft"

# Masking rate 37.5%, attention
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_attention' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention"

# Masking rate 37.5%, ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "ffn"


# Masking rate 37.5%, add_layernorm
python -u run.py \
--run_name $run_name \
--is_training True \
--model $model \
--root ./data/electricity/ \
--data_path electricity.csv \
--data custom \
--mask_rate 0.375 \
--sequence_len 96 \
--label_len 0 \
--prediction_len 0 \
--encoder_input 321 \
--decoder_input 321 \
--output_size 321 \
--gpt_layers 3 \
--batch_size 16 \
--model_dimension 768 \
--description 'Experiment_Electricity_0375_1703_add_layernorm' \
--iterations 1 \
--mlp 1 \
--learning_rate 0.001 \
--lora True \
--components "add_layernorm"

# Masking rate 37.5%, attention_add_layernorm
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_attention_add_layernorm' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_add_layernorm"

# Masking rate 37.5%, attention_ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_attention_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_ffn"

# Masking rate 37.5%, ffn_add_layernorm
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_ffn_add_layernorm' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "ffn_add_layernorm"

# Masking rate 37.5%, all
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_1703_attention_add_layernorm_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_add_layernorm_ffn"

