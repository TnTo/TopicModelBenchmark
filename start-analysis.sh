#!/bin/bash
sudo systemctl start docker
git submodule update --init --remote --recursive
docker-compose -f docker-compose-analysis.yml build
docker-compose -f docker-compose-analysis.yml up