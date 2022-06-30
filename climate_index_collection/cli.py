from pathlib import Path

import click

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import ClimateIndexFunctions


# This should go into a separate submodule
def run_complete_workflow(
    input_path=None, model_name=None, index_name=None, output_path=None
):
    """Run complete workflow for a single given model and index

    Parameters
    ----------
    input_path: str | path
        Input data path.
    model_name: str
        Name of model data source.
    index_name: str
        Name of the index.
    output_path: str | path
        Path to the output data.

    Returns
    -------
    path
        Path to the output data file.

    """
    output_file = Path(output_path) / f"{model_name}_{index_name}.nc"

    index_function = ClimateIndexFunctions[index_name].value
    model_data_set = load_data_set(data_path=input_path, data_source_name=model_name)

    index_data_set = index_function(model_data_set)

    index_data_set.compute().to_netcdf(output_file)

    return output_file


@click.command()
@click.option("--input-path", default=".", help="Path to the input data.")
@click.option("--output-path", default=".", help="Path to the output data.")
@click.option(
    "--model-names",
    default=",".join(tuple(VARNAME_MAPPING.keys())),
    help=(
        f"Space separated list of the input data sources "
        f'Defaults to: "{",".join(tuple(VARNAME_MAPPING.keys()))}"'
    ),
)
@click.option(
    "--index-names",
    default=",".join(ClimateIndexFunctions.get_all_names()),
    help=(
        f"Space separated list of all indices you want to calculate. "
        f'Defaults to: "{",".join(ClimateIndexFunctions.get_all_names())}"'
    ),
)
def run(input_path, output_path, model_names, index_names):
    """Command line interface to the climate index collection."""
    click.echo(f"Will look for input data in: {input_path}")
    click.echo(f"Will write outputs to: {output_path}")
    click.echo(f"Will calculate indices for: {model_names}")
    click.echo(f"Will calculate following indices: {index_names}")

    model_names = model_names.split(",")
    index_names = index_names.split(",")

    for model_name in model_names:
        for index_name in index_names:
            output_file = run_complete_workflow(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                index_name=index_name,
            )
            print(f"Done for: {output_file}")
