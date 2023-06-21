start=$(date +%s)
cd .. &&

python FedAverage.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --model='resnet18_IN' --mode='LDP' --round=100 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=3 --E=1
python FedAverage.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --model='resnet18_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=3 --E=1
python FedAverage.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --model='resnet18_IN' --mode='CDP' --round=30 --epsilon=2 --sr=1 --lr=5e-2 --flr=1e-1 --physical_bs=8 --E=1
python FedAverage.py --data='cifar10' --nclient=100 --nclass=10 --ncpc=2 --model='resnet18_IN' --mode='CDP' --round=60 --epsilon=8 --sr=1 --lr=5e-2 --flr=1e-1 --physical_bs=8 --E=1

python log/show.py --E=1 --data='cifar10'
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.


