#!/bin/bash
set -x
sudo apt-get install git -y
sudo rm -rf /tmp/PDMChartMaker
cd /tmp
git clone https://github.com/321core/PDMChartMaker
cd PDMChartMaker
sudo apt-get install python-pip --upgrade
sh install-pymodules.sh

