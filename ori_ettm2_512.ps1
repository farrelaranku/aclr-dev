# Daftar perintah yang akan dijalankan
$commands = @(
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_96_ori --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_96_ori --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_96_ori --model Reformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_96_ori --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_96_ori --model Transformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",


    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_192_ori --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_192_ori --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_192_ori --model Reformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_192_ori --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_192_ori --model Transformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",

    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_336_ori --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_336_ori --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_336_ori --model Reformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_336_ori --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_336_ori --model Transformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",

    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_720_ori --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_720_ori --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_720_ori --model Reformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_720_ori --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_ori.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ettm2_96_720_ori --model Transformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20"

)

# Eksekusi setiap perintah dan hentikan jika ada error
foreach ($cmd in $commands) {
    Write-Host "Running: $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error encountered, stopping script."
        exit $LASTEXITCODE
    }
}

Write-Host "All models have been trained successfully."