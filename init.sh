#!/bin/bash

# Create empty directory on AWS ephemeral drive
sudo mkdir -p /mnt/data
sudo chown carnd.carnd /mnt/data
# rm -r /mnt/data/*.p

sudo dd if=/dev/zero of=/mnt/swapfile.img bs=1024 count=16M
sudo chmod 0600 /mnt/swapfile.img 
sudo mkswap /mnt/swapfile.img 
sudo swapon /mnt/swapfile.img
cat /proc/swaps 
free -m 
