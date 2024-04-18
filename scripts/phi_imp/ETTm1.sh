model=PhiImputer
run_name="first_run"

# Masking rate 12.5%
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --phi_layers 8 \
  --batch_size 16 \
  --model_dimension 2560 \
  --description 'Experiment_ETTm1_0125_0903' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 25.0%
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --phi_layers 8 \
  --batch_size 16 \
  --model_dimension 2560 \
  --description 'Experiment_ETTm1_025_0903' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 37.5%
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --phi_layers 8 \
  --batch_size 16 \
  --model_dimension 2560 \
  --description 'Experiment_ETTm1_0375_0903' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 50.0%
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --phi_layers 8 \
  --batch_size 16 \
  --model_dimension 2560 \
  --description 'Experiment_ETTm1_05_0903' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True
