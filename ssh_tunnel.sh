#!/bin/bash
#
# Source: https://github.com/spadin/behavioral-cloning/blob/master/bin/enable-ssh-tunnel.sh

if [ -z "$1" ]; then
  echo "Usage: sh enable-ssh-tunnel.sh [ip-address]"
  echo ""
  exit 1
fi
ssh -f -N -M -S /tmp/carnd.sock -L 4567:localhost:4567 carnd@$1 2> /dev/null
