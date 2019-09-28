source ~/miniconda3/bin/activate ~/miniconda3/envs/transformer_env
mkdir $2/model
CUDA_VISIBLE_DEVICES=$1 nohup python ~/NMT/fairseq/train.py ~/2nd_year/ontrac/how2/transformer/data_fethi/iwslt19_bpe30k_en_pt --arch transformer_wmt_en_de \
-s en \
-t pt \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 \
--lr 0.0005 \
--min-lr 1e-09 \
--max-update 500000 \
--dropout 0.3 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 2048 \
--save-dir $2/model/ > $2/training_log 2>&1  &
