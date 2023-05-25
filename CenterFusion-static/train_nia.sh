# # cd src
# # train
# python src/main.py \
#     ddd \
#     --exp_id centerfusion \
#     --shuffle_train \
#     --train_split train \
#     --val_split val \
#     --val_intervals 10 \
#     --nuscenes_att \
#     --velocity \
#     --batch_size 64 \
#     --lr 2.5e-4 \
#     --num_epochs 100 \
#     --lr_step 50 \
#     --save_all \
#     --gpus 3 \
#     --not_rand_crop \
#     --flip 0.5 \
#     --shift 0.1 \
#     --pointcloud \
#     --radar_sweeps 3 \
#     --pc_z_offset 0.0 \
#     --pillar_dims 1.0,0.2,0.2 \
#     --max_pc_dist 80.0 \
#     --print_iter 50 \
#     --num_workers 8 \
#     --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/all \
#     --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/train
#     # --freeze_backbone \
#     # --resume \

# cd src
# train
python src/main.py \
    ddd \
    --exp_id centerfusion \
    --shuffle_train \
    --train_split train \
    --val_split val \
    --val_intervals 1 \
    --nuscenes_att \
    --velocity \
    --batch_size 1 \
    --lr 2.5e-4 \
    --num_epochs 100 \
    --lr_step 50 \
    --save_all \
    --gpus 0 \
    --not_rand_crop \
    --flip 0.5 \
    --shift 0.1 \
    --pointcloud \
    --radar_sweeps 3 \
    --pc_z_offset 0.0 \
    --pillar_dims 1.0,0.2,0.2 \
    --max_pc_dist 80.0 \
    --print_iter 50 \
    --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/sample \
    --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/sample/train
    # --freeze_backbone \
    # --resume \