#!/bin/bash

# Create empty directory on AWS ephemeral drive
sudo mkdir -p /mnt/data
sudo chown carnd.carnd /mnt/data
# rm -r /mnt/data/*.p
