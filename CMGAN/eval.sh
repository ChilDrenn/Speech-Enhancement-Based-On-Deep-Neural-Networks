python evaluation.py \
    --test_dir "path of test set, for example, ./DEMAND_16KHz/test" \
    --model_path "path of evaluated model" \
    --save_dir "path for saving results" \
    --attn1 "attention block: se/cbam/eca/simam/None"

# Best model evaluation: 
python evaluation.py \
    --test_dir ./DEMAND_16KHz/test \
    --model_path ./best_ckpt/CBAM_pesq \
    --save_dir ./CBAM_pesq \
    --attn1 cbam

python evaluation.py \
    --test_dir ./DEMAND_16KHz/test \
    --model_path ./best_ckpt/CBAM_ssnr \
    --save_dir ./CBAM_ssnr \
    --attn1 cbam

python evaluation.py \
    --test_dir ./DEMAND_16KHz/test \
    --model_path  ./best_ckpt/MR_STFT \
    --save_dir ./MR_STFT \
    --attn1 None
