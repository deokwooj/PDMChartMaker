#!/bin/bash
set -x
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
sudo pip install --upgrade pip
sudo apt-get install git -y
sudo rm -rf /tmp/PDMChartMaker
cd /tmp
git clone https://github.com/321core/PDMChartMaker
cd PDMChartMaker
sudo apt-get install python-pip --upgrade
sh install-pymodules.sh
mv /tmp/PDMChartMaker .
