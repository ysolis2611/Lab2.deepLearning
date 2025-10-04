# Usando Apple Silicon GPU (MPS)

model_name=iTransformer


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_24 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 8 \
  --learning_rate 1e-05 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --lradj cosine \
  --warmup_epochs 2 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0001 \
  --grad_clip 3.0 


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_48 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 8 \
  --learning_rate 1e-05 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --lradj cosine \
  --warmup_epochs 2 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0001 \
  --grad_clip 3.0 


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 8 \
  --learning_rate 2e-05 \
  --train_epochs 15 \
  --patience 7 \
  --batch_size 64 \
  --lradj cosine \
  --warmup_epochs 3 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0001 \
  --grad_clip 5.0 


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 8 \
  --learning_rate 2e-05 \
  --train_epochs 15 \
  --patience 7 \
  --batch_size 64 \
  --lradj cosine \
  --warmup_epochs 3 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0001 \
  --grad_clip 5.0 


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --n_heads 16 \
  --learning_rate 5e-05 \
  --train_epochs 20 \
  --patience 10 \
  --batch_size 32 \
  --lradj cosine \
  --warmup_epochs 5 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0002 \
  --grad_clip 7.0 


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --n_heads 16 \
  --learning_rate 5e-05 \
  --train_epochs 30 \
  --patience 15 \
  --batch_size 32 \
  --lradj cosine \
  --warmup_epochs 5 \
  --use_gpu \
  --gpu 0 \
  --devices mps \
  --itr 1 \
  --weight_decay 0.0003 \
  --grad_clip 10.0 

