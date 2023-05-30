cd ./src/server

CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset mnist \
 --nat_pn_backbone lenet5 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \


CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistS \
 --nat_pn_backbone lenet5 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \


 CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistA \
 --nat_pn_backbone lenet5 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \


 CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset medmnistC \
 --nat_pn_backbone lenet5 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \


CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset cifar10 \
 --nat_pn_backbone res18 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \


CUDA_VISIBLE_DEVICES=1 python fedavg.py --dataset svhn \
 --nat_pn_backbone res18 \
 --batch_size 64 \
 --loss_name bayessian \
 --loss_log_prob_weight 0.0001 \
 --loss_entropy_weight 0.0 \
 --loss_embeddings_weight 0.0 \
 --finetune_in_the_end 10 \
 --global_epoch 100 \
 --local_epoch 1 \
 --stop_grad_logp true \
 --stop_grad_embeddings false \
 --server_cuda 1 \
 --client_cuda 1 \
 --local_lr 0.0001 \
 --momentum 0.0 \
 --eval_test 0 \
 --save_metrics 0 \
 --save_fig 0 \
 --save_log 0 \