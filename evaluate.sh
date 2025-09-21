python main.py \
  --seed 42 \
  -b 32 \
  --d_hidden 512 \
  --d_ff 1024 \
  --d_embedding 1024 \
  --num_heads 8 \
  --num_layers 4 \
  --test_size 0.2 \
  --early_stop 20 \
  --dropout 0 \
  --model saves/20250920_111008.pth


#python main.py --seed 42 -b 32 --d_hidden 512 --d_ff 1024 --d_embedding 1024 --num_heads 8 --num_layers 4 --test_size 0.2 --early_stop 20 --dropout 0 --model saves/20250920_111008.pth
