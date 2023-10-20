# bash
artists=$1
device=$2

for artist in ${artists}; do
   python glaze_origin.py --clean_data_dir=../wikiart/preprocessed_data/${artist}/clean/train/ \
                          --trans_data_dir=../wikiart/preprocessed_data/${artist}/trans/train/transNum${TRANS_NUM}_seed${SEED} \
                          --p=${glaze_p} \
                          --alpha=${glaze_alpha} \
                          --glaze_iters=${glaze_iters} \
                          --lr=${glaze_lr} \
                          --device=${device}

  python glaze_attack.py --clean_data_dir=../wikiart/preprocessed_data/${artist}/clean/train/ \
                         --trans_data_dir=../wikiart/preprocessed_data/${artist}/trans/train/transNum${TRANS_NUM}_seed${SEED} \
                         --pur_eps=${pur_eps} \
                         --pur_lr=${pur_lr} \
                         --pur_iters=${pur_iters} \
                         --pur_alpha=${pur_alpha} \
                         --pur_noise=${pur_noise} \
                         --device=${device} \
                         --neg_feed=${neg_feed} \
                         --adv_para=${adv_dir} \
                         --pur_para=${pur_dir}

done
