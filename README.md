# PCGL-GAN
Towards SAR-to-Optical Image Translation via Perception Correlative Learning and Global-Local Feature Collaborative

## Prerequisites
- python 3.10
- torch 2.3.1
- CUDA 11.8
- dominate
- visdom

## Trian
```
python train.py --dataroot [] --name [] --model [] --gpu_ids 0 --lambda_spatial 10 --lambda_gradient 0 --attn_layers 4,7,9 --loss_mode cos --gan_mode lsgan --display_port 8097 --direction AtoB --patch_size 64
```

## Test
```
python test.py --dataroot [] --checkpoints_dir ./checkpoints --name [] --model [] --num_test []
```

## Acknowledgments
This code heavily borrowes from [CUT](https://github.com/JunlinHan/DCLGAN),[F/LSeSim](https://github.com/lyndonzheng/F-LSeSim), and [KAN-CUT](https://github.com/amaha7984/KAN-CUT).

## Note
The full version of the paper will be uploaded after acceptance.
