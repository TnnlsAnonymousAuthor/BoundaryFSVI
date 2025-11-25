# SVHN
python run_inference.py --dataset svhn --cost_type md_cost --ctx_dataset none
python run_inference.py --dataset svhn --cost_type md_cost --ctx_dataset svhn_augmentation --beta 0.0005
python run_inference.py --dataset svhn --cost_type md_cost --ctx_dataset eurosat --beta 0.0005
python run_inference.py --dataset svhn --cost_type md_cost --ctx_dataset cifar100_augmentation --beta 0.0005
python run_gan_inference.py --dataset svhn --cost_type md_cost --beta 0.0005
python run_inference.py --dataset svhn --cost_type kl_cost --ctx_dataset svhn_augmentation --beta 1
python run_inference.py --dataset svhn --cost_type kl_cost --ctx_dataset eurosat --beta 1
python run_inference.py --dataset svhn --cost_type kl_cost --ctx_dataset cifar100_augmentation --beta 1
python run_gan_inference.py --dataset svhn --cost_type kl_cost --beta 1

# CIFAR10
python run_inference.py --dataset cifar10 --cost_type md_cost --ctx_dataset none
python run_inference.py --dataset cifar10 --cost_type md_cost --ctx_dataset cifar10_augmentation --beta 0.00001
python run_inference.py --dataset cifar10 --cost_type md_cost --ctx_dataset eurosat --beta 0.00001
python run_inference.py --dataset cifar10 --cost_type md_cost --ctx_dataset cifar100_augmentation --beta 0.00001
python run_gan_inference.py --dataset cifar10 --cost_type md_cost --beta 0.00001
python run_inference.py --dataset cifar10 --cost_type kl_cost --ctx_dataset cifar10_augmentation --beta 1
python run_inference.py --dataset cifar10 --cost_type kl_cost --ctx_dataset eurosat --beta 1
python run_inference.py --dataset cifar10 --cost_type kl_cost --ctx_dataset cifar100_augmentation --beta 1
python run_gan_inference.py --dataset cifar10 --cost_type kl_cost --beta 1


