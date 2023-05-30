# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../src

# Perform detection and evaluation
# python test.py ddd \
#     --dataset nia \
#     --gpus 0 \
#     --val_split test_normal \
#     --run_dataset_eval \
#     --nuscenes_att \
#     --velocity \
#     --print_iter 100 \
#     --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/sample \
#     --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/sample/test/temp \
#     --load_model /data/kimgh/CenterFusion-custom/CenterFusion-static/result/all/train/model_63.pth \
    # --pointcloud \


python test.py ddd \
    --dataset nia \
    --gpus 0 \
    --val_split test_normal \
    --run_dataset_eval \
    --nuscenes_att \
    --velocity \
    --print_iter 100 \
    --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/selectsub1 \
    --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/selectsub1/test/normal_nopc_epoch22 \
    --load_model /data/kimgh/CenterFusion-custom/CenterFusion-static/result/selectsub1/train/model_22.pth





