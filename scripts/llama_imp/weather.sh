model=LlamaImputer
run_name="final_run_check"

# Masking rate 12.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_Weather_0125_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 25.0%
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_Weather_025_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 37.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_Weather_0375_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True


# Masking rate 50.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_Weather_05_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True
