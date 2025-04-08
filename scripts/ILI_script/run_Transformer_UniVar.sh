# export CUDA_VISIBLE_DEVICES=0

python -u run_ori.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_Transformer_36_24 \
  --model Transformer \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 3

python -u run_ori.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_Transformer_36_36 \
  --model Transformer \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 3

python -u run_ori.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_Transformer_36_48 \
  --model Transformer \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 3

python -u run_ori.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_Transformer_36_60 \
  --model Transformer \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 3