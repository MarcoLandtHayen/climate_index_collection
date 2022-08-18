FROM pangeo/pangeo-notebook:2022.07.27 AS buildstage

WORKDIR /source
USER root

RUN apt-get update
RUN apt-get install -y git

COPY . .
# ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.1
RUN python -m pip wheel .
# RUN python -m pip install clim*.whl

FROM pangeo/pangeo-notebook:2022.07.27 AS app

COPY --from=buildstage /source/clim*.whl .

RUN python -m pip install clim*.whl