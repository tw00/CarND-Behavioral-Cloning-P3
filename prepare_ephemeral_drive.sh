#!/bin/bash

# Create empty directory on AWS ephemeral drive
sudo mkdir -p /mnt/data
sudo chown carnd.carnd /mnt/data
sudo mkdir -p /mnt/models
sudo chown carnd.carnd /mnt/models
cp -r data/* /mnt/data/
# rm -r /mnt/data/*.p

# http://askubuntu.com/questions/178712/how-to-increase-swap-space
sudo dd if=/dev/zero of=/mnt/swapfile.img bs=1024 count=16M
sudo chmod 0600 /mnt/swapfile.img 
sudo mkswap /mnt/swapfile.img 
sudo swapon /mnt/swapfile.img
cat /proc/swaps 
free -m 

#
source activate carnd-term1
./run_postprocess.py --dataset dataset1_udacity
#./run_postprocess.py --dataset dataset2_twe_one_lap
#./run_postprocess.py --dataset dataset3_ssz_one_lap
./run_postprocess.py --dataset dataset4_beta_sim
./run_postprocess.py --dataset dataset5_beta_backwards
./run_postprocess.py --dataset dataset6_curve2A
./run_postprocess.py --dataset dataset7_curve2B
