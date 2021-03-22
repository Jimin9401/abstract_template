
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


NGPU=0
CUDA_VISIBLE_DEVICES=$NGPU python main.py \
    --root data \
    --seed $S \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_step 1;
