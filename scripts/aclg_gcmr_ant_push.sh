REWARD_SHAPING=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

# The hyperparameters associated with method A are marked with backslash (\\**\\)

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntPush-v1" \
--reward_shaping ${REWARD_SHAPING} \
--algo aclg \
--correction_type OPC \
--osp_delta 0 \
--osp_delta_update_rate 0 \
--rollout_exp_w 0 \
\
\
--ctrl_gp_lambda 0.0001 \
--ctrl_gp_obs_noise 2 \
--ctrl_gp_start_step 20000 \
--landmark_loss_coeff 0.1 \
--sampling_mix_mode HR \
--traj_max_num 800 \
--weighting_alpha 0.1 \
\
\
--ctrl_osrp_lambda 0 \
--version "${REWARD_SHAPING}_HG2P" \
--goal_loss_coeff 0.02 \
--delta 3.0 \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--landmark_sampling fps \
--n_landmark_coverage 60 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 60 \
--man_buffer_size $((10**6)) \
--ctrl_buffer_size $((10**6)) \
--ctrl_rew_scale 1.5 \
--man_rew_scale 0.1 \
--r_margin_pos 2.0 \
--r_margin_neg 2.4