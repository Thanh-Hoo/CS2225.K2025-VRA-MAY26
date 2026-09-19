#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate IDEA
cd /13994058190/WYH/IDEA_PUBLIC
python train.py --config_file /13994058190/WYH/IDEA_PUBLIC/configs/RGBNT201/IDEA.yml
