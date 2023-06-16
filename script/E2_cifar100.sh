start=$(date +%s)
cd ../transfer &&

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2

python FedTransfer.py --data='cifar100' --nclient=100 --naclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='CDP' --round=20 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2

python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='clip' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2
python FedTransfer.py --data='cifar100' --nclient=100 --nclass=100 --ncpc=2 --encoder='simclr' --model='linear_model_DN_IN' --mode='LDP' --round=20 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=2


cd .. &&
python log/show.py --E=2 --data='cifar100'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.