export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0003 --steps 1e7 --batch 16 --train_freq 1 --target_update 500 --node 128 --hidden_n 1 --final_eps 0.05 --learning_starts 1000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.1 --clip_rewards
python run_qnet.py --algo C51 --env BreakoutNoFrameskip-v4 --learning_rate 0.0003 --steps 1e7 --batch 16 --train_freq 1 --target_update 500 --node 128 --hidden_n 1 --final_eps 0.05 --learning_starts 1000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.1 --min 0 --max 30 --clip_rewards
python run_qnet.py --algo QRDQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0003 --steps 1e7 --batch 16 --train_freq 1 --target_update 500 --node 128 --hidden_n 1 --final_eps 0.05 --learning_starts 1000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.1 --n_support 200 --delta 0.1 --clip_rewards
python run_qnet.py --algo IQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0003 --steps 1e7 --batch 16 --train_freq 1 --target_update 500 --node 128 --hidden_n 1 --final_eps 0.05 --learning_starts 1000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.1 --n_support 64 --delta 0.1 --clip_rewards