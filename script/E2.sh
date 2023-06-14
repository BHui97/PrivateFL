cd ../transfer &&
#python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8

#python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8

#python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='LDP' --round=40 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8
python FedTransfer.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --encoder='resnext' --model='linear_model_DN_IN' --mode='CDP' --round=40 --epsilon=8 --sr=1 --lr=1e-2 --flr=1e-1 --physical_bs=8