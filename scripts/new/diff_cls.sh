# bash
artists=$1
device=$2

for artist in ${artists}; do
  # test_data_dir="${artist}/${test_data}/test/500_transNum24_seed0/"
  python diffusion-classifier/eval_prob_adaptive.py \
       --artist="${artist}" \
       --test_data="${test_data}" \
       --adv_para=${adv_dir} \
       --pur_para=${pur_dir} \
       --ft_step=${step} \
       --trans_num=${TRANS_NUM} \
       --device="${device}" \
       --manual_seed=${SEED}

done