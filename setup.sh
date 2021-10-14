conda create --name topicmodelbenchmark python mamba
conda activate topicmodelbenchmark
mamba env update --file ./adso/environment.yml --name topicmodelbenchmark
pip install ./adso
mamba env update --file ./analysis/environment.yml --name topicmodelbenchmark