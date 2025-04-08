python -u run_singlemodU.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_E3k_singlemod_36_24 \
  --model B6autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --d_model 256 \
  --itr 1 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints1/


python -u run_singlemodU.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_E3k_singlemod_36_36 \
  --model B6autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --d_model 256 \
  --itr 1 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints1/



python -u run_singlemodU.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_E3k_singlemod_36_48 \
  --model B6autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --d_model 256 \
  --itr 1 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints2/


python -u run_singlemodU.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_E3k_singlemod_36_60 \
  --model B6autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --dropout 0.1 \
  --d_model 256 \
  --itr 1 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints3/
