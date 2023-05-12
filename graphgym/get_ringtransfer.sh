#!/bin/bash

# define an array of numbers
layers=(5 10 15 20 30 40 50 60)

# stage=stack
# layer=gcnconv

# stage=delay_gnn
# layer=my_gcnconv

stage=k_gnn
layer=my_gcnconv

d=256

# loop through the array and print each number
for L in "${layers[@]}"
do
  dir="results/og_cfg/${stage}_L=${L}"
  python main.py --cfg /Users/beng/Documents/pytorch_geometric/graphgym/configs/pre_dec_22/ring_transfer.yaml --repeat 3 gnn.layers_mp $L gnn.layer_type $layer gnn.stage_type $stage out_dir $dir
done