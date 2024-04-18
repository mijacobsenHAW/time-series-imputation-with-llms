model=LlamaImputer

# Masking rate 12.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_ETTh2_0125_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True


# Masking rate 25.0%,

python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_ETTh2_025_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True


# Masking rate 37.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_ETTh2_0375_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True


# Masking rate 50.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --llama_layers 8 \
  --batch_size 16 \
  --model_dimension 4096 \
  --description 'Experiment_ETTh2_05_2902' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True
