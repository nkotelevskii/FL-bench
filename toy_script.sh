#!/bin/bash

# n_classes_list=(3 5 10 15 20 30 40 50 75 100 150 200)
n_classes_list=(200)
random_seeds=(40 41 42 43 44)

for n_classes in "${n_classes_list[@]}"; do
  cd ./data/utils
  python run.py -d toy_noisy --iid 1 -cn 1 --toy_noisy_classes "${n_classes}"
  cd ../..

  for seed in "${random_seeds[@]}"; do
  echo "Processing ${n_classes}, seed ${seed}..."
    cd ./src/server
    CUDA_VISIBLE_DEVICES=0 python fedavg.py \
    --dataset toy_noisy \
    --nat_pn_backbone 2nn \
    --finetune_in_the_end 0 \
    --eval_test 0 \
    --save_metrics 0 \
    --save_fig 0 \
    --save_log 0 \
    --global_epoch 1 \
    --local_epoch 100 \
    --loss_name bayessian \
    --stop_grad_logp false \
    --stop_grad_embeddings false \
    --loss_log_prob_weight 0.0 \
    --loss_entropy_weight 0.0 \
    --loss_embeddings_weight 0.0 \
    --seed "${seed}" \
    --local_lr 0.003 \
    --server_cuda 1 \
    --client_cuda 1 \
    
    cd ../..
  done
done
