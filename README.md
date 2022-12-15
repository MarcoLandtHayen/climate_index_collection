# Climate Index Collection

[![Build Status](https://github.com/MarcoLandtHayen/climate_index_collection/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/climate_index_collection/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/mlandthayen/climate_index_collection?label=DockerHub)](https://hub.docker.com/r/mlandthayen/climate_index_collection/tags)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7440574.svg)](https://doi.org/10.5281/zenodo.7440574)


Collection of climate indices derived from climate model outputs.


_See [notebooks/Tutorial.ipynb](notebooks/Tutorial.ipynb) for details._


## Development

For now, we're developing in the Pangeo notebook containter. More details: https://github.com/pangeo-data/pangeo-docker-images

To start a JupyterLab within this container, run
```shell
$ docker pull pangeo/pangeo-notebook:2022.07.27
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work pangeo/pangeo-notebook:2022.07.27 jupyter lab --ip=0.0.0.0
```
and open the URL starting on `http://127.0.0.1...`.

Then, open a Terminal within JupyterLab and run
```shell
$ python -m pip install -e .
```
to have a local editable installation of the package.

## Container Image

There's a container image: https://hub.docker.com/r/mlandthayen/climate_index_collection

### Use with Docker

You can use it wherever Docker is installed by running:
```shell
$ docker pull mlandthayen/climate_index_collection:<tag>
$ docker run --rm -v $PWD:/work -w /work mlandthayen/climate_index_collection:<tag> climate_index_collection_run --help
```
Here, `<tag>` can either be `latest` or a more specific tag.

### Use with Singularity

You can use it wherever Singularity is installed by essentially running:
```shell
$ singularity pull --disable-cache --dir "${PWD}" docker://mlandthayen/climate_index_collection:<tag>
$ singularity run climate_index_collection_<tag>.sif climate_index_collection_run --help
```
Here, `<tag>` can either be `latest` or a more specific tag.

_Note_ that for NESH, it's currently necessary to
- specify the version of singularity to use, and
- to make sure to bind mount various parts of the file system explicitly.

So the full call on NESH would look like:
```shell
$ module load singularity/3.5.2
$ singularity pull --disable-cache --dir "${PWD}" docker://mlandthayen/climate_index_collection:<tag>
$ singularity run -B /sfs -B /gxfs_work1 -B ${PWD}:/work --pwd /work climate_index_collection_<tag>.sif climate_index_collection_run --help
```

## Release Procedure

A release will contain the specific version of the package (taken care of automatically) and the CSV file created with the full data.

1. _**Draft a release:**_ Go to https://github.com/MarcoLandtHayen/climate_index_collection/releases/new and draft a new release (don't publish yet).

2. _**Prepeare data:**_ For the commit in `main` for which the release is planned, pull the container on NESH (see above) and run:
```
$ singularity run -B /sfs -B /gxfs_work1 -B ${PWD}:/work --pwd /work climate_index_collection_<tag>.sif climate_index_collection_run --input-path <path_to_full_data>
```

3. _**Attach data:**_ Attach the CSV file to the drafted release.

4. _**Publish:**_ By clicking on the `Publish release` button.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
