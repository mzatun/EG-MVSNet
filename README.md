# EG-MVSNet
The official implement of EG-MVSNet by PyTorch.

## Aerial Building MVS Dataset
Download Link: [Baidu NetDisk](https://pan.baidu.com/s/1me2lrSkskNiveJGMOJu5JQ?pwd=mufv)

Due to the limitation of uploading size, we have split the dataset.zip into four files by the below command:
```
split -b 3072mb MVS_Aerial_Building.zip MVS_Aerial_Building_ 
```

The overall zip file can be merged by the following command:
```
cat MVS_Aerial_Building_* > MVS_Aerial_Building.zip
```

## Code
### Train
1. In ```train.py```, set ```mode``` to ```train```, set ```model``` to ```egnet```
   
2. In ```train.py```, set ```trainpath``` to your train data path ```YOUR_PATH/dataset/train```, set ```testpath``` to your train data path ```YOUR_PATH/dataset/test```

3. Train EG-MVSNet (RTX 3090 24G):
```
python train.py
```

### Test
1. In ```train.py```, set ```testpath``` to your train data path ```YOUR_PATH/dataset/test```,
   set ```loadckpt``` to your model path ```./checkpoints/xx/xxx.ckpt```, set depth sample number ```numdepth```.

2. Run EG-MVSNet (RTX 3090 24G):
```
python train.py 
```
