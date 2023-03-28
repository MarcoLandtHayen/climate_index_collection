# Climate Index Collection

[![Build Status](https://github.com/MarcoLandtHayen/climate_index_collection/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/climate_index_collection/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/mlandthayen/climate_index_collection?label=DockerHub)](https://hub.docker.com/r/mlandthayen/climate_index_collection/tags)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7440574.svg)](https://doi.org/10.5281/zenodo.7440574)


Collection of climate indices derived from climate model outputs.


## Quickstart: Using the climate indices

The resulting climate-index time series are published to Zenodo under the DOI [10.5281/zenodo.7436144](https://doi.org/10.5281/zenodo.7436144) and you can obtain the index timeseries by manually downloading the file `climate_indices.csv` from this dataset.

You can also use [pooch](https://www.fatiando.org/pooch/latest/) to obtain the published index time series programmatically:
```python
import pooch

climate_indices_file = pooch.retrieve(
    url="doi:10.5281/zenodo.7436144/climate_indices.csv",
    known_hash=None,
)
```
With `climate_indices_file` containing the path to the CSV file either resulting from the code above or from manually setting it to the location of the manually downloaded data, we recommend using [Pandas](https://pandas.pydata.org/docs/) for reading the data:
```python
import pandas as pd

climate_indices = pd.read_csv(climate_indices_file)
```
This results in a dataframe with the following structure:
```python
print(climate_indices)
```
```
       model  year  month   index                         long_name     value
0       FOCI     1      1  SAM_ZM  southern_annular_mode_zonal_mean -0.295492
1       FOCI     1      2  SAM_ZM  southern_annular_mode_zonal_mean  0.530890
2       FOCI     1      3  SAM_ZM  southern_annular_mode_zonal_mean  1.684005
3       FOCI     1      4  SAM_ZM  southern_annular_mode_zonal_mean  1.409169
4       FOCI     1      5  SAM_ZM  southern_annular_mode_zonal_mean  0.984511
...      ...   ...    ...     ...                               ...       ...
695647  CESM   999      8      NP                     north_pacific -0.210202
695648  CESM   999      9      NP                     north_pacific  0.206541
695649  CESM   999     10      NP                     north_pacific -0.331067
695650  CESM   999     11      NP                     north_pacific  0.487844
695651  CESM   999     12      NP                     north_pacific -0.782657

[695652 rows x 6 columns]
```
To apply statistics or to plot all indices, you can apply standard modifications provided by Pandas. Calculating, e.g., the standard deviation of all indices amounts to
```python
print(climate_indices.groupby(["model", "index"])[["value"]].std())
```
```
model  index
CESM   AMO            0.109084
       ENSO_12        0.603481
       ENSO_3         0.881938
       ENSO_34        0.933544
       ENSO_4         0.909434
       NAO_PC         1.000042
       NAO_ST         1.554596
       NP             0.569459
       PDO_PC         1.000042
...    ...            ...
FOCI   AMO            0.128715
       ENSO_12        0.342368
       ENSO_3         0.600242
       ENSO_34        0.759405
       ENSO_4         0.923854
       NAO_PC         1.000042
       NAO_ST         1.459676
       NP             0.644014
       PDO_PC         1.000042
...    ...            ...
Name: value, dtype: float64
```

## Quickstart: Reproducing the dataset

The Python package in this repository can be installed using [`pip`](https://pip.pypa.io/en/stable/getting-started/#install-a-package-from-github):
```shell
$ python -m pip install git+https://github.com/MarcoLandtHayen/climate_index_collection.git@v2022.12.15.1
```
The data from which the indices have been calculated are published under the DOI [10.5281/zenodo.7060385](https://doi.org/10.5281/zenodo.7060385). After downloading the data to, e.g., `./cicmod_data/`, you can run the command line version of this package by
```shell
$ climate_index_collection_run --input-path ./cicmod_data/ --output-path .
```
which will create a file `climate_indices.csv`.

Please see either `$ climate_index_collection_run --help` on the command line, or the tutorial notebook in [notebooks/Tutorial.ipynb](notebooks/Tutorial.ipynb) for more details.


## Development

For now, we're developing in the Pangeo notebook container. More details: https://github.com/pangeo-data/pangeo-docker-images

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
