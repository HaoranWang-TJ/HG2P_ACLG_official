REWARD_SHAPING=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

# The hyperparameters associated with method A are marked with backslash (\\**\\)

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "PointMaze-v1" \
--reward_shaping ${REWARD_SHAPING} \
--algo aclg \
--correction_type OPC \
--osp_delta 0 \
--osp_delta_update_rate 0 \
--rollout_exp_w 0 \
\
\
--ctrl_gp_lambda 0.001 \
--ctrl_gp_obs_noise 2 \
--ctrl_gp_start_step 20000 \
--landmark_loss_coeff 10 \
--sampling_mix_mode HR \
--hr_buffer_max_size 120000 \
--weighting_alpha 0.1 \
\
\
--ctrl_osrp_lambda 0 \
--version "${REWARD_SHAPING}_HG2P" \
--goal_loss_coeff 20 \
--delta 2.0 \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--landmark_sampling fps \
--n_landmark_coverage 60 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 60 \
--adj_factor 0.7 \
--clip_v -10