model=GptImputer
run_name="final_run_2003"

# --train_epochs 30 \
#
python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --model_dimension 768 \
  --batch_size 16 \
  --description 'Experiment_Weather_0125_2003' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True

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
  --model_dimension 768 \
  --batch_size 16 \
  --description 'Experiment_Weather_025_2003' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --model_dimension 768 \
  --batch_size 16 \
  --description 'Experiment_Weather_0375_2003' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/weather/ \
  --data_path weather.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --label_len 0 \
  --prediction_len 0 \
  --encoder_input 21 \
  --decoder_input 21 \
  --output_size 21 \
  --gpt_layers 3 \
  --model_dimension 768 \
  --batch_size 16 \
  --description 'Experiment_Weather_05_2003' \
  --iterations 1 \
  --learning_rate 0.001 \
  --lora True
