# bash
export TRANS_NUM=24
export SEED=0

# glaze hyperparameters
export glaze_p=0.05
export glaze_alpha=30
export glaze_iters=500
export glaze_lr=0.01

# purify hyperparameters
export pur_eps=0.1
export pur_lr=0.01
export pur_iters=3000
export pur_alpha=0.1
export pur_noise=0.1

# diff hyperparameters
export batch_size=8
export grad_accum=1
export step=500

# export adv_dir="adv_adapt_p${glaze_p}_alpha${glaze_alpha}_iter${glaze_iters}_lr${glaze_lr}"
# export pur_dir="pur_adapt_eps${pur_eps}-iters${pur_iters}-lr${pur_lr}-pur_alpha${pur_alpha}-noise${pur_noise}-neg${neg_feed}"
export adv_dir="adv_p${glaze_p}_alpha${glaze_alpha}_iter${glaze_iters}_lr${glaze_lr}"
export pur_dir="pur_eps${pur_eps}-iters${pur_iters}-lr${pur_lr}-pur_alpha${pur_alpha}-noise${pur_noise}-neg${neg_feed}"
# adapt: above, adv_pur, should adjust hyperp of consist loss

all_artists='raphael-kirchner camille-pissarro pyotr-konchalovsky childe-hassam paul-cezanne claude-monet albrecht-durer eugene-boudin'

device1="cuda:0"
device2="cuda:1"
device3="cuda:2"
device4="cuda:3"


artists1="raphael-kirchner camille-pissarro"
artists2="pyotr-konchalovsky childe-hassam"
artists3="paul-cezanne claude-monet"
artists4="albrecht-durer eugene-boudin"



 ## glaze and pur all

bash scripts/new/adv_pur.sh "${artists1}" ${device1} & bash scripts/new/adv_pur.sh "${artists2}" ${device2} & bash scripts/new/adv_pur.sh "${artists3}" ${device3} & bash scripts/new/adv_pur.sh "${artists4}" ${device4} & wait
 sleep 1m

 finetune SD model
 for artist in $all_artists; do
     export OUTPUT_DIR="../stable_diffusion_models/${artist}/clean_${step}/"
     bash scripts/new/finetune_sd.sh "../wikiart/preprocessed_data/${artist}/clean/train/"

     export OUTPUT_DIR="../stable_diffusion_models/${artist}/${adv_dir}_${step}_transNum${TRANS_NUM}_seed${SEED}/"
     bash scripts/new/finetune_sd.sh "../wikiart/preprocessed_data/${artist}/${adv_dir}/train/transNum${TRANS_NUM}_seed${SEED}"

     export OUTPUT_DIR="../stable_diffusion_models/${artist}/${pur_dir}_${step}_transNum${TRANS_NUM}_seed${SEED}/"
     bash scripts/new/finetune_sd.sh "../wikiart/preprocessed_data/${artist}/${pur_dir}/train/transNum${TRANS_NUM}_seed${SEED}"

 done

 # test
bash scripts/new/diff_generate.sh "${artists1}" ${device1} & bash scripts/new/diff_generate.sh "${artists2}" ${device2} & bash scripts/new/diff_generate.sh "${artists3}" ${device3} & bash scripts/new/diff_generate.sh "${artists4}" ${device4} & wait

# clip test
python clip_classifier.py \
       --all_artists="${all_artists}" \
       --adv_para=${adv_dir} \
       --pur_para=${pur_dir} \
       --ft_step=${step} \
       --trans_num=${TRANS_NUM} \
       --manual_seed=${SEED}
