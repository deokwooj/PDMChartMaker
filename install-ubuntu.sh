#!/bin/bash
set -x
PWD = `pwd`
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev git -y
sudo pip install --upgrade pip
sudo rm -rf /tmp/PDMChartMaker
cd /tmp
git clone https://github.com/deokwooj/PDMChartMaker
cd PDMChartMaker
sh install-pymodules.sh
mv /tmp/PDMChartMaker PWD
