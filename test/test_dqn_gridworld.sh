export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env GridWorld"
LR="--learning_rate 0.0002"
TRAIN="--steps 5e4 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.01 --learning_starts 1000 --gamma 0.9 --buffer_size 1e5 --exploration_fraction 0.2"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--time_scale 1 --capture_frame_rate 30"
OPTIMIZER="--optimizer adamw"
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24:32' python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
