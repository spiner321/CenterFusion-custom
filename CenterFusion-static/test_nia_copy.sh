# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd ../src

# Perform detection and evaluation
python src/test.py ddd \
    --dataset nia \
    --gpus 2 \
    --val_split test_norm \
    --run_dataset_eval \
    --nuscenes_att \
    --velocity \
    --pointcloud \
    --print_iter 100 \
    --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/all \
    --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/test/temp \
    --load_model /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/train/model_18.pth \


# python src/test.py ddd \
#     --exp_id centerfusion \
#     --dataset nia \
#     --val_split test_norm \
#     --run_dataset_eval \
#     --nuscenes_att \
#     --velocity \
#     --gpus 1 \
#     --pointcloud \
#     --radar_sweeps 3 \
#     --max_pc_dist 60.0 \
#     --pc_z_offset -0.0 \
#     --flip_test \
#     --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/all \
#     --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/test/normal_epoch18 \
#     --load_model /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/train/model_18.pth \





