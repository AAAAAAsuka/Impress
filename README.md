# Impress
This is the official repository for "[Do Imperceptible Perturbations Really Prevent Unauthorized Data Usage in Diffusion-based Image Generation? (need to add link here)]()", the paper has been accepted by NeurIPS 2023

## Environment Setup
First, our code requires the environment listed in ```requirements.txt``` to be installed:
```bash
pip install -r requirements.txt
```

## Generate Experiment Data
We have conducted experiments with the Glaze and Photoguard methods on subsets of the wikiart dataset and the Helenface dataset respectively. Below is how to generate the protected target images used in the experiment.

For the Glaze method, the wikiart dataset needs to be downloaded first. After the dataset is downloaded, generate the experiment data using the following command:
```bash
python wikiart_preprocessing.py --wikiart_dir=your_wikiart_dir --exp_data_dir=your_exp_data_dir
```
The generated experimental data will be stored in the ```your_exp_data_dir/${artist}/clean/train/``` directory. Where, ```artist``` is the author of the selected artwork.

For the Photoguard method, the experimental data used can be downloaded through this link.

## Quick Start
For the Glaze method, to quickly start our experiment, please execute the following command:
```bash
bash scripts/new/test_all.sh
```
Next, we will introduce each step in detail.

## Adding Protective Noise to Images
For the Glaze method, it is first necessary to generate the style-transferred protected images:
```bash
python style_transfer.py --exp_data_dir=your_exp_data_dir --artist=artist
```

Then, execute the following code to add protective noise to the images:
```bash
python glaze_origin.py --clean_data_dir=[exp_data_dir]/${artist}/clean/train/ \
                        --trans_data_dir=[exp_data_dir]/preprocessed_data/${artist}/trans/train/transNum24_seed0 \
                        --p=${glaze_p} \
                        --alpha=${glaze_alpha} \
                        --glaze_iters=${glaze_iters} \
                        --lr=${glaze_lr} \
                        --device=${device}
```
Below is the explanation of input hyperparameters:
* ```glaze_p```: Hyperparameter p in the Glaze method, can be seen as a perturbation budget, default is 0.05.
* ```glaze_alpha```: Hyperparameter \( \alpha \) in the Glaze method, used to balance the two loss items, default is 30.
* ```glaze_iters```: The number of iterations to perform the Glaze method, default is 500.
* ```glaze_lr```: The learning rate used in the Glaze method, default is 0.01.

Photoguard to be completed.

## Execute Impress
For data protected by Glaze, to use Impress to remove protective noise, execute the following code:
```bash
python glaze_pur.py --clean_data_dir=[exp_data_dir]/${artist}/clean/train/ \
                    --trans_data_dir=[exp_data_dir]/${artist}/trans/train/transNum24_seed0 \
                    --pur_eps=${pur_eps} \
                    --pur_lr=${pur_lr} \
                    --pur_iters=${pur_iters} \
                    --pur_alpha=${pur_alpha} \
                    --pur_noise=${pur_noise} \
                    --device=${device} \
                    --adv_para=${adv_para} \
                    --pur_para=${pur_para}
```
Below is the explanation of input hyperparameters:
* ```pur_eps```: Hyperparameter p in our Impress method, can be considered as a perturbation budget, default is 0.1.
* ```pur_lr```: The learning rate used in our Impress method, default is 0.01.
* ```pur_iters```: The number of iterations in our Impress method, default is 3000.
* ```pur_alpha```: Hyperparameter \( \alpha \) in our Impress method, used to balance the two loss items, default is 0.1.
* ```pur_noise```: The intensity of the Gaussian noise initially added to the images in our Impress method, default is 0.1.
* ```adv_para```: Hyperparameters used in executing Glaze, used for establishing storage directories. Format is ```"adv_p${glaze_p}_alpha${glaze_alpha}_iter${glaze_iters}_lr${glaze_lr}"```
* ```pur_para```: Hyperparameters used in executing Impress, used for establishing storage directories. Format is ```"pur_eps${pur_eps}-iters${pur_iters}-lr${pur_lr}-pur_alpha${pur_alpha}-noise${pur_noise}"```

## (Glaze) Finetune Stable Diffusion Model
After generating the initial target images, images protected by Glaze, and images purified by Impress, we need to use these images to finetune the Stable Diffusion model separately:
```bash
accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1-base' \
  --train_data_dir=${TRAIN_DIR} \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=${batch_size} \
  --gradient_accumulation_steps=${grad_accum} \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=${step} \
  --learning_rate=5e-6 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --enable_xformers_memory_efficient_attention
```
Where, ```TRAIN_DIR``` is the storage path of images used as the finetune dataset. For explanations of other parameters, please see https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image.

## Using Stable Diffusion to Generate Images
For Glaze, we need to use the model finetuned in the previous step to generate images. Please execute:
```bash
python glaze_test.py \
    --test_data_dir="../wikiart/preprocessed_data/${artist}/clean/test/" \
    --save_dir=your_generate_images_save_dir \
    --checkpoint=your_finetuned_SDmodel_savedir\
    --diff_steps=100 \
    --device=${test_device} 
```

## Evaluation
For Glaze, to calculate evaluate metrics, please execute:

```bash
#clip classifier 
python clip_classifier.py \
       --all_artists="${all_artists}" \
       --adv_para=${adv_para} \
       --pur_para=${pur_para} \
       --ft_step=${step} \
       --trans_num=24 \
       --manual_seed=0

# diffusion classifier
  python diffusion-classifier/eval_prob_adaptive.py \
       --artist="${artist}" \
       --test_data="${test_data}" \
       --adv_para=${adv_dir} \
       --pur_para=${pur_dir} \
       --ft_step=${step} \
       --trans_num=24 \
       --device="${device}" \
       --manual_seed=0

```
Where, ```adv_para``` and ```pur_para``` are the same as the inputs when executing Impress, ```step``` is the step number when finetuning the model. For the CLIP classifier, ```all_artists``` represents all artists to be tested, required to be entered as a string, and separated by spaces. For the Diffusion classifier, ```test_data``` represents the type of data to be tested, it can be ```clean```, ```adv```, or ```pur```.