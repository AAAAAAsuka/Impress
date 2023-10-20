
# adv parameters
export attack_type="l2"
export pg_eps=16
export pg_step_size=1
export pg_iters=200
export pg_grad_reps=10
export pg_eta=1
# pur parameters
export neg_feed=-1
export pur_eps=0.15
export pur_iters=75
export pur_lr=0.005
export pur_alpha=0.01
export pur_noise=0.05

# generate parameters
export test_guidance=7.5
export test_diff_steps=50

export CUDA_LAUNCH_BLOCKING=1
 export prompt="a person in an airplane"
#export prompt="a person in Europe"
# export prompt="a person with a red hat"
# export prompt="a person on the restaurant"

 adv
 python pg_mask_diff_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=0 --device="cuda:0" & \
 python pg_mask_diff_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=1 --device="cuda:1" & \
 python pg_mask_diff_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=2 --device="cuda:2" & \
 python pg_mask_diff_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=3 --device="cuda:3" & wait

# pur
python pg_mask_pur_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=0 --device="cuda:0" \
      --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise & \
python pg_mask_pur_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=1 --device="cuda:1" \
      --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise & \
python pg_mask_pur_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=2 --device="cuda:2" \
      --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise & \
python pg_mask_pur_helen.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=3 --device="cuda:3" \
      --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise & wait

python pg_generate.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=0 --device="cuda:0" \
       --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise \
       --test_guidance=$test_guidance --test_diff_steps=$test_diff_steps --prompt="${prompt}" & \
python pg_generate.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=1 --device="cuda:1" \
       --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise \
       --test_guidance=$test_guidance --test_diff_steps=$test_diff_steps --prompt="${prompt}" & \
python pg_generate.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=2 --device="cuda:2" \
       --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise \
       --test_guidance=$test_guidance --test_diff_steps=$test_diff_steps --prompt="${prompt}" & \
python pg_generate.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=3 --device="cuda:3" \
       --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise \
       --test_guidance=$test_guidance --test_diff_steps=$test_diff_steps --prompt="${prompt}" & wait


python pg_metric.py --attack_type=$attack_type --pg_eps=$pg_eps --pg_step_size=$pg_step_size --pg_iters=$pg_iters --pg_grad_reps=$pg_grad_reps --pg_eta=$pg_eta --parallel_index=3 --device="cuda:3" \
      --neg_feed=$neg_feed --pur_eps=$pur_eps --pur_iters=$pur_iters --pur_lr=$pur_lr --pur_alpha=$pur_alpha --pur_noise=$pur_noise \
      --test_guidance=$test_guidance --test_diff_steps=$test_diff_steps --prompt="${prompt}"