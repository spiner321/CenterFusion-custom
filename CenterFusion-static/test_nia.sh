export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd ../src

## Perform detection and evaluation
python src/test.py ddd \
    --dataset nia \
    --gpus 0,1,2,3 \
    --val_split test_norm \
    --run_dataset_eval \
    --nuscenes_att \
    --velocity \
    --pointcloud \
    --data_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/data/sample \
    --save_dir /data/kimgh/CenterFusion-custom/CenterFusion-static/result/test/pretrain \
    --load_model ./models/centerfusion.pth \
    # --resume \