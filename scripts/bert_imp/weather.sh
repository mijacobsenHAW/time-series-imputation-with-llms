model=BertImputer
run_name="final_run_check"

# Masking rate 12.5%
python -u run.py \
  --run_name $run_name \
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
  --bert_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_0125_1103' \
  --iterations 1 \
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
  --bert_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_025_1103' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True

 Masking rate 37.5%
python -u run.py \
  --run_name $run_name \
  --is_training False \
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
  --bert_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_0375_1103' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True

# Masking rate 50.0%
python -u run.py \
  --run_name $run_name \
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
  --bert_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Weather_05_1103' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True
