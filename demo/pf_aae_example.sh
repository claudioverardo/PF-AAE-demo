#!/bin/bash

#############################################################################
#                           PF-AAE: tracking demo                           #
#         please read the preparatory steps in README.md beforehand         #
#############################################################################

# generate a 'poisson' sequence (cf. auto_pose/pf/pfilter_aae.py)
pf_generate_sequences aae/cracker \
    -params "{'seq': 'poisson', 'fps': 20, 'tlen': 10, 'rate': 0.5, 'mu': 30, 'sigma': 30, 'seed': 500}" \
    -back_img "2007_000783.jpg" \
    -occl "{'p': 0.3, 'p_drop': 0.3, 'size_percent': 0.05}" \
    -inter [4.2,[0,0,1],180,0]

# the sequence has been generated in a subdirectory of the filtering directory with name:
# "seq=poisson_fps=20.00_tlen=10.00_rate=0.50_mu=30.00_sigma=30.00_seed=500_inters=[4.2]_[[0.0, 0.0, 1.0]]_[180.0]_[0.0]_occl={'p': 0.3, 'p_drop': 0.3, 'size_percent': 0.05}"
# the following instructions start 3 training experiments, whose results and logs will be saved in the above folder

# tracking experiment without AAE_TL resampling and uniform resampling
pf_tracking_sequences \
    -aae "aae/cracker" -N 50 \
    -noise "{'type': 'unif-norm', 'sigma': 0.2}" \
    -gamma 100 -n_eff 1.0 \
    -aae_res "{'type': 'none'}" \
    -seq "seq=poisson_fps=20.00_tlen=10.00_rate=0.50_mu=30.00_sigma=30.00_seed=500_inters=[4.2]_[[0.0, 0.0, 1.0]]_[180.0]_[0.0]_occl={'p': 0.3, 'p_drop': 0.3, 'size_percent': 0.05}" \
    -fps 20

# tracking experiment with uniform resampling
pf_tracking_sequences \
    -aae "aae/cracker" -N 50 \
    -noise "{'type': 'unif-norm', 'sigma': 0.2}" \
    -gamma 100 -n_eff 1.0 \
    -aae_res "{'type': 'unif', 'aae_resampling_proportion': 0.1}" \
    -seq "seq=poisson_fps=20.00_tlen=10.00_rate=0.50_mu=30.00_sigma=30.00_seed=500_inters=[4.2]_[[0.0, 0.0, 1.0]]_[180.0]_[0.0]_occl={'p': 0.3, 'p_drop': 0.3, 'size_percent': 0.05}" \
    -fps 20

# tracking experiment with AAE_TL resampling
pf_tracking_sequences \
    -aae "aae/cracker" -aae_tl "aae_tl/cracker" -N 50 \
    -noise "{'type': 'unif-norm', 'sigma': 0.2}" \
    -gamma 100 -n_eff 1.0 \
    -aae_res "{'type': 'aae-tl', 'aae_resampling_proportion': 0.1, 'aae_resampling_knn': 300}" \
    -seq "seq=poisson_fps=20.00_tlen=10.00_rate=0.50_mu=30.00_sigma=30.00_seed=500_inters=[4.2]_[[0.0, 0.0, 1.0]]_[180.0]_[0.0]_occl={'p': 0.3, 'p_drop': 0.3, 'size_percent': 0.05}" \
    -fps 20