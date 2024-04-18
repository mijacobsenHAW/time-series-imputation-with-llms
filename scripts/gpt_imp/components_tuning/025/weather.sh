model=GptImputer
run_name="components_tuning_3_layers_025"

# Masking rate 25%, no fine-tuning
#python -u run.py \
#  --run_name $run_name \
#  --is_training True \
#  --model $model \
#  --root ./data/weather/ 
#  --data_path weather.csv \
#  --data custom \
#  --mask_rate 0.25 \
#  --sequence_len 96 \
#  --label_len 0 \
#  --prediction_len 0 \
#  --encoder_input 21 \
#  --decoder_input 21 \
#  --output_size 21 \
#  --gpt_layers 3 \
#  --batch_size 16 \
#  --model_dimension 768 \
#  --description 'Experiment_Weather025_1703' \
#  --iterations 1 \
#  --mlp 1 \
#  --learning_rate 0.001 \
#  --lora True \
#  --components "no_ft"

# Masking rate 25%, attention
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_attention' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention"

# Masking rate 25%, ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "ffn"


# Masking rate 25%, add_layernorm
python -u run.py \
--run_name $run_name \
--is_training True \
--model $model \
--root ./data/weather/ \
--data_path weather.csv \
--data custom \
--mask_rate 0.25 \
--sequence_len 96 \
--label_len 0 \
--prediction_len 0 \
--encoder_input 21 \
--decoder_input 21 \
--output_size 21 \
--gpt_layers 3 \
--batch_size 16 \
--model_dimension 768 \
--description 'Experiment_Weather_025_1703_add_layernorm' \
--iterations 1 \
--mlp 1 \
--learning_rate 0.001 \
--lora True \
--components "add_layernorm"

# Masking rate 25%, attention_add_layernorm
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_attention_add_layernorm' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_add_layernorm"

# Masking rate 25%, attention_ffn
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_attention_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_ffn"

# Masking rate 25%, ffn_add_layernorm
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_ffn_add_layernorm' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "ffn_add_layernorm"

# Masking rate 25%, all
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1703_attention_add_layernorm_ffn' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --components "attention_add_layernorm_ffn"

