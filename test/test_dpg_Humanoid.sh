export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env Humanoid-v4"
LR="--learning_rate 0.0003"
TRAIN="--steps 5e6 --buffer_size 1e6 --batch 256 --target_update_tau 5e-3 --learning_starts 50000"
MODEL="--node 256 --hidden_n 2"
OPTIONS="--logdir log/"
OPTIMIZER="--optimizer adam"

python  run_dpg.py --algo TD3 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo SAC $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD7 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
