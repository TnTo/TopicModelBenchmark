FROM mambaorg/micromamba
COPY environment.yml /environment.yml
COPY adso_environment.yml /adso_environment.yml
COPY docker_environment.yml /docker_environment.yml
USER root
RUN apt-get update && apt-get install -y mongo-tools && apt-get clean
USER micromamba
RUN micromamba install -y -n base -f /docker_environment.yml && \
    micromamba install -y -n base -f /adso_environment.yml && \
    micromamba install -y -n base -f /environment.yml && \
    micromamba clean --all --yes
WORKDIR /sacred
COPY cmd.sh /cmd.sh
CMD ["/cmd.sh"]