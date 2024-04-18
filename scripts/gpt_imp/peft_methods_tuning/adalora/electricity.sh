model=GptImputer
run_name="peft_methods_tuning/adalora"

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.125 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0125_2703_adalora' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --peft_type "ada_lora"

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.25 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_025_2703_adalora' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --peft_type "ada_lora"

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.375 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_0375_2703_adalora' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --peft_type "ada_lora"

python -u run.py \
  --run_name $run_name \
  --is_training True \
  --model $model \
  --root ./data/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --mask_rate 0.5 \
  --sequence_len 96 \
  --prediction_len 0 \
  --label_len 0 \
  --encoder_input 321 \
  --decoder_input 321 \
  --output_size 321 \
  --gpt_layers 3 \
  --batch_size 16 \
  --model_dimension 768 \
  --description 'Experiment_Electricity_05_2703_adalora' \
  --iterations 1 \
  --mlp 1 \
  --learning_rate 0.001 \
  --lora True \
  --peft_type "ada_lora"