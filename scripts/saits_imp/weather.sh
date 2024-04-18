# config gathered from https://github.com/WenjieDu/SAITS/blob/main/configs/Electricity_SAITS_base.ini
model=SaitsImputer

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
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_Weather_0125_2902' \
  --iterations 1 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

# Masking rate 25.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_Weather_025_2902' \
  --iterations 1 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

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
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_Weather_0375_2902' \
  --iterations 1 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

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
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_Weather_05_2902' \
  --iterations 1 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

