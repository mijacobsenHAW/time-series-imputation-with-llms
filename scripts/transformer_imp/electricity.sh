model=TransformerImputer

# Masking rate 12.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 128 \
  --d_ff 128 \
  --des 'Experiment_Electricity_0125_2802' \
  --iterations 3 \
  --top_k 5 \
  --learning_rate 0.001

# Masking rate 25.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 128 \
  --d_ff 128 \
  --des 'Experiment_Electricity_025_2802' \
  --iterations 3 \
  --top_k 5 \
  --learning_rate 0.001

# Masking rate 37.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 128 \
  --d_ff 128 \
  --des 'Experiment_Electricity_0375_2802' \
  --iterations 3 \
  --top_k 5 \
  --learning_rate 0.001

# Masking rate 50.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 128 \
  --d_ff 128 \
  --des 'Experiment_Electricity_05_2802' \
  --iterations 3 \
  --top_k 5 \
  --learning_rate 0.001