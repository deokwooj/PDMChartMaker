#!/bin/bash
set -x
INSTALL_DIR=`pwd`
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev libfreetype6-dev git -y
sudo pip install --upgrade pip
sudo rm -rf /tmp/PDMChartMaker
cd /tmp
git clone https://github.com/deokwooj/PDMChartMaker
cd PDMChartMaker
sh install-pymodules.sh
mv /tmp/PDMChartMaker $INSTALL_DIR
