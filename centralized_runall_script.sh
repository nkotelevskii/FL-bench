cd ./src/server

CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset noisy_mnist \
 --nat_pn_backbone lenet5 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.00001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 0 \
 --global_epoch 1 \
 --local_epoch 10 \
 --batch_size 64 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0003 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \
 --save_prefix centralized

# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset noisy_cifar100 \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.00001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 30 \
#  --batch_size 128 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.0003 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --save_prefix centralized

# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset noisy_cifar100 \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 30 \
#  --batch_size 128 \
#  --stop_grad_logp false \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.0003 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --save_prefix centralized


#  CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset fmnist \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
# --save_prefix centralized


# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistS \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
# --save_prefix centralized


#  CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistA \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
# --save_prefix centralized


#  CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistC \
#  --nat_pn_backbone lenet5 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
# --save_prefix centralized

# 0.0001 - logprobweight
# 0.0003 - lr

#  CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset cifar10 \
#  --nat_pn_backbone res18 \
#  --batch_size 128 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.0001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.0003 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
#  --save_prefix centralized


# CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset svhn \
#  --nat_pn_backbone res18 \
#  --loss_name bayessian \
#  --loss_log_prob_weight 0.00001 \
#  --loss_entropy_weight 0.0 \
#  --loss_embeddings_weight 0.0 \
#  --finetune_in_the_end 0 \
#  --global_epoch 1 \
#  --local_epoch 10 \
#  --stop_grad_logp true \
#  --stop_grad_embeddings false \
#  --server_cuda 1 \
#  --client_cuda 1 \
#  --local_lr 0.001 \
#  --eval_test 0 \
#  --save_metrics 0 \
#  --save_fig 0 \
#  --save_log 0 \
# --save_prefix centralized
