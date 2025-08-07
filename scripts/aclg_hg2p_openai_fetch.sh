ENV=$1
TIMESTEPS=$2
GPU=$3
SEED=$4

# The hyperparameters associated with method A are marked with backslash (\\**\\)

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--absolute_goal \
--delta 2.0 \
--env_name ${ENV} \
--reward_shaping "sparse" \
--algo aclg \
--correction_type OPC \
--osp_delta 0 \
--osp_delta_update_rate 0 \
--rollout_exp_w 0 \
\
\
--ctrl_gp_lambda 0.001 \
--ctrl_gp_obs_noise 0.2 \
--ctrl_gp_start_step 10000 \
--landmark_loss_coeff 10 \
--sampling_mix_mode HR \
--hr_buffer_max_size 20000 \
--weighting_alpha 0.1 \
\
\
--ctrl_osrp_lambda 0 \
--goal_loss_coeff 0 \
--seed ${SEED} \
--max_timesteps  ${TIMESTEPS} \
--manager_propose_freq 5 \
--landmark_sampling fps \
--n_landmark_coverage 60 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 60 \
--ctrl_noise_sigma 0.3 \
--man_noise_sigma 0.2 \
--train_ctrl_policy_noise 0.2 \
--train_man_policy_noise 0.2 \
--ctrl_rew_scale 1.0 \
--man_rew_scale 0.01 \
--r_margin_pos 0.1 \
--r_margin_neg 0.12 \
--close_thr 0.2 \
--clip_v -15 \
--goal_thr -5 \
--version "sparse_HG2P"
