#python ../FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=60 --epsilon=2 --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=8
#python ../FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1 --lr=5e-3 --flr=1e-1 --physical_bs=8
#python ../FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=60 --epsilon=2 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8
#python ../FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=1 --lr=1e-1 --flr=1e-1 --physical_bs=8

#python ../FedAverage.py --data='fashionmnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=150 --epsilon=8 --sr=0.3 --lr=1e-1 --flr=1e-1 --physical_bs=8
#python ../FedAverage.py --data='fashionmnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=60 --epsilon=2 --sr=0.3 --lr=1e-1 --flr=1e-1 --physical_bs=8
#python ../FedAverage.py --data='fashionmnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='CDP' --round=150 --epsilon=8 --sr=1.0 --lr=1e-2 --flr=1e-1 --physical_bs=8
#python ../FedAverage.py --data='fashionmnist' --nclient=100 --nclass=10 --ncpc=2 --model='mnist_fully_connected_IN' --mode='LDP' --round=60 --epsilon=2 --sr=0.3 --lr=2e-2 --flr=1e-1 --physical_bs=8