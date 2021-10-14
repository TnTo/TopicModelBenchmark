#!/bin/bash
sudo systemctl start docker
git submodule update --init --remote --recursive
docker-compose -f docker-compose-experiment.yml build
docker-compose -f docker-compose-experiment.yml up
# docker-compose -f docker-compose-experiment.yml logs -f sacred
