# Usando Apple Silicon GPU (MPS)

model_name=iTransformer


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_24 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_48 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./iTransformer_datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 64 \
  --use_gpu \
  --gpu 0 \
  --device mps \
  --itr 1

