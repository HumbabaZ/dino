[DINO](https://github.com/facebookresearch/dino)

[Correlation Clustering of Organoid Images](https://github.com/JannikPresberger/Correlation_Clustering_of_Organoid_Images/tree/main/code-twin-networks)

# Correlation Clustering of Organoid Images with DINO

## DINO training with organoid images
```
srun torchrun --nproc_per_node=4 main_dino.py \
    --arch vit_small \
    --data_path /projects/p_rep_learn_2/datasets \
    --output_dir $WORKSPACE_DIR/output \
    --saveckp_freq 20
```

## Twin network training using pretrained DINO checkpoints
```
srun python siamese_network.py \
    --model-name dino-p0.0 \
    --model-dir $WORKSPACE_DIR/models \
    --input-type dino \
    --embedding-dimension 384 \
    --augment 0.0 \
    --data-dir $TRAIN_DATA_PATH \
    --val-data-dir $VAL_DATA_PATH \
    --dino-checkpoint $DINO_CHECKPOINT \
    --dino-arch vit_small \
    --batch-size 32 \
    --val-batch-size 64 \
    --lr 0.002 \
    --total-steps 30000 \
    --steps-per-epoch 100 \
    --override
```

## Correlation clustering
