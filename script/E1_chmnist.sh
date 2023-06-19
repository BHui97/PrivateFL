start=$(date +%s)
cd .. &&
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet_IN' --mode='CDP' --round=20 --epsilon=2 --sr=0.8 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet_IN' --mode='CDP' --round=50 --epsilon=8 --sr=0.8 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet_IN' --mode='LDP' --round=10 --epsilon=2 --sr=0.8 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='chmnist' --nclient=40 --nclass=8 --ncpc=2 --model='alexnet_IN' --mode='LDP' --round=30 --epsilon=8 --sr=0.8 --lr=1e-1 --flr=1e-1 --physical_bs=8 --E=1

python log/show.py --E=1 --data='mnist'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


