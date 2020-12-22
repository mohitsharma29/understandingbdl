for (( seed=4; seed <=4; seed++ ))
do
    #python experiments/train/run_swag.py --data_path=$DATAPATHT --epochs=$EPOCHS --dataset=$DATASET --save_freq=$EPOCHS \
    #  --model=$MODEL --lr_init=${LR} --wd=${WD} --swag --swag_start=$SWAG_START --swag_lr=${SWAG_LR} --cov_mat --use_test \
    #  --dir=${BASEDIR}/swag_${seed} --seed=$seed
    CUDA_VISIBLE_DEVICES=1 python experiments/train/run_swag.py --epochs=300 --dataset=CIFAR10 --save_freq=300 --model=PreResNet20 --lr_init=0.1 \
     --wd=3e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test --dir=/media/data_dump/Mohit/bayesianML/models/iter${seed}/ \
     --data_path=/media/data_dump/Mohit/bayesianML/cifar-normal/cifar-10-batches-py/ --seed=$seed
done
