# # cd src
# # train
python src/main.py \
    ddd \
    --exp_id centerfusion \
    --shuffle_train \
    --train_split train \
    --val_split val \
    --val_intervals 50 \
    --nuscenes_att \
    --velocity \
    --batch_size 72 \
    --num_workers 8 \
    --lr 2.5e-4 \
    --num_epochs 300 \
    --lr_step 50 \
    --save_all \
    --gpus 3 \
    --not_rand_crop \
    --flip 0.5 \
    --shift 0.1 \
    --pointcloud \
    --radar_sweeps 3 \
    --pc_z_offset 0.0 \
    --pillar_dims 1.0,0.2,0.2 \
    --max_pc_dist 80.0 \
    --print_iter 500 \
    --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/selectsub1 \
    --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/selectsub1/train \
    --load_model /data/kimgh/CenterFusion-custom/CenterFusion-static/result/selectsub1/train/model_last.pth \
    --resume
#     # --freeze_backbone \



# cd src
# train
# python src/main.py \
#     ddd \
#     --exp_id centerfusion \
#     --shuffle_train \
#     --train_split train \
#     --val_split val \
#     --val_intervals 1 \
#     --nuscenes_att \
#     --velocity \
#     --batch_size 1 \
#     --lr 2.5e-4 \
#     --num_epochs 100 \
#     --lr_step 50 \
#     --save_all \
#     --gpus 0 \
#     --not_rand_crop \
#     --flip 0.5 \
#     --shift 0.1 \
#     --pointcloud \
#     --radar_sweeps 3 \
#     --pc_z_offset 0.0 \
#     --pillar_dims 1.0,0.2,0.2 \
#     --max_pc_dist 80.0 \
#     --print_iter 50 \
#     --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/sample \
#     --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/sample/train
    # --freeze_backbone \
    # --resume \