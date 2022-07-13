# Climate Index Collection

[![Build Status](https://github.com/MarcoLandtHayen/climate_index_collection/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/climate_index_collection/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/climate_index_collection)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![pypi](https://img.shields.io/pypi/v/climate_index_collection.svg)](https://pypi.org/project/climate_index_collection)
[![Documentation Status](https://readthedocs.org/projects/climate_index_collection/badge/?version=latest)](https://climate_index_collection.readthedocs.io/en/latest/?badge=latest)


Collection of climate indices derived from climate model outputs.


_See [notebooks/Tutorial.ipynb](notebooks/Tutorial.ipynb) for details._


## Development

For now, we're developing in the Pangeo notebook containter. More details: https://github.com/pangeo-data/pangeo-docker-images

To start a JupyterLab within this container, run
```shell
$ docker pull pangeo/pangeo-notebook:2022.05.10
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work pangeo/pangeo-notebook:2022.05.10 jupyter lab --ip=0.0.0.0
```
and open the URL starting on `http://127.0.0.1...`.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
