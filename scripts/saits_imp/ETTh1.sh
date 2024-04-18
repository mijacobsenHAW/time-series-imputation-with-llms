# config gathered from https://github.com/WenjieDu/SAITS/blob/main/configs/Electricity_SAITS_base.ini
model=SaitsImputer

# Masking rate 12.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --mask_rate 0.125 \
  --data ETTh1 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 7 \
  --dropout 0.2 \
  --decoder_input 7 \
  --output_size 7 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_ETTh1_0125_2902' \
  --iterations 3 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

# Masking rate 25.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --mask_rate 0.25 \
  --data ETTh1 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --dropout 0.2 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_ETTh1_025_2902' \
  --iterations 3 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

# Masking rate 37.5%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --mask_rate 0.375 \
  --dropout 0.2 \
  --data ETTh1 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_ETTh1_0375_2902' \
  --iterations 3 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

# Masking rate 50.0%
python -u run.py \
  --run_name "first_run_2702" \
  --is_training True \
  --model $model \
  --root ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --mask_rate 0.50 \
  --dropout 0.2 \
  --data ETTh1 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --encoder_input 7 \
  --decoder_input 7 \
  --output_size 7 \
  --factor 3 \
  --batch_size 16 \
  --model_dimension 256 \
  --d_ff 128 \
  --v 64 \
  --k 64 \
  --n_heads 4 \
  --mit_weight 1 \
  --ort_weight 1 \
  --des 'Experiment_ETTh1_05_2902' \
  --iterations 3 \
  --learning_rate 0.001 \
  --diagonal_attention_mask True

