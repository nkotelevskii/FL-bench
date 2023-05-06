cd ./src/server

CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset femnist \
 --nat_pn_backbone lenet5 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0 \
 --loss_entropy_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 250 \
 --local_epoch 5 \
 --stop_grad_logp false \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.001 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0


# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset mnist \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.001 \
#  --loss_entropy_weight 0.0 \
#  --finetune_in_the_end 10 \
#  --global_epoch 10 \
#  --local_epoch 5 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings true \
#  --server_cuda 0 \
#  --client_cuda 0 \
#  --local_lr 0.0001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0