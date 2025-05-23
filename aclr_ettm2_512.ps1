# Daftar perintah yang akan dijalankan
$commands = @(
    # "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_96_noACLR_b512 --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_96_withACLR_b512 --use_aclr --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    # "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_192_noACLR_b512 --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_192_withACLR_b512 --use_aclr --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    # "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_336_noACLR_b512 --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_336_withACLR_b512 --use_aclr --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    # "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_720_noACLR_b512 --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20",
    "python -u run_open_net.py --is_training 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id ETTm2_96_720_withACLR_b512 --use_aclr --model B6iFast --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --batch_size 512 --checkpoint ./checkpoints/ --train_epochs 20"
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