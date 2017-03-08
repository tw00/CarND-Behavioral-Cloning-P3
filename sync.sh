#!/bin/bash

mkdir -p models
rsync --delete -avzbe ssh carnd@$1:/mnt/models/ ./models --backup-dir=../_old/
