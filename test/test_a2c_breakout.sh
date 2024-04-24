export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0003"
TRAIN="--steps 5e5 --worker 32 --batch 32 --mini_batch 256 --gamma 0.99 --lamda 0.95 --ent_coef 1e-3"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--val_coef 0.6 --gae_normalize"

OPTIMIZER="--optimizer rmsprop"

python run_pg.py --algo A2C $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
