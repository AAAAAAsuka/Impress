all_artists_test=$1
test_device=$2
for artist in $all_artists_test; do

      python glaze_test.py \
         --test_data_dir="../wikiart/preprocessed_data/${artist}/clean/test/" \
         --save_dir="../wikiart/preprocessed_data/${artist}/clean/test/${step}/" \
         --checkpoint="../stable_diffusion_models/${artist}/clean_${step}/"\
         --diff_steps=100 \
         --device=${test_device}

     python glaze_test.py \
        --test_data_dir="../wikiart/preprocessed_data/${artist}/clean/test/" \
        --save_dir="../wikiart/preprocessed_data/${artist}/${adv_dir}/test/${step}_transNum${TRANS_NUM}_seed${SEED}/" \
        --checkpoint="../stable_diffusion_models/${artist}/${adv_dir}_${step}_transNum${TRANS_NUM}_seed${SEED}/"\
        --diff_steps=100 \
        --device=${test_device}

    python glaze_test.py \
       --test_data_dir="../wikiart/preprocessed_data/${artist}/clean/test/" \
       --save_dir="../wikiart/preprocessed_data/${artist}/${pur_dir}/test/${step}_transNum${TRANS_NUM}_seed${SEED}/" \
       --checkpoint="../stable_diffusion_models/${artist}/${pur_dir}_${step}_transNum${TRANS_NUM}_seed${SEED}/"\
       --diff_steps=100 \
       --device=${test_device} 

done