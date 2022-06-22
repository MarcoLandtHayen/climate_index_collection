import click


from climate_index_collection.indices import ClimateIndexFunctions
from climate_index_collection.data_loading import VARNAME_MAPPING


@click.command()
@click.option("--input-path", default=".", help="Path to the input data.")
@click.option("--output-path", default=".", help="Path to the output data.")
@click.option(
    "--model-names", default=",".join(tuple(VARNAME_MAPPING.keys())), 
    help=(
        f'Space separated list of the input data sources '
        f'Defaults to: "{",".join(tuple(VARNAME_MAPPING.keys()))}"'
    )
)
@click.option(
    "--index-names", default=",".join(ClimateIndexFunctions.get_all_names()), 
    help=(
        f'Space separated list of all indices you want to calculate. '
        f'Defaults to: "{",".join(ClimateIndexFunctions.get_all_names())}"'
    )
)
def run(input_path, output_path, model_names, index_names):
    """Command line interface to the climate index collection."""
    click.echo(f"Will look for input data in: {input_path}")
    click.echo(f"Will write outputs to: {output_path}")
    click.echo(f"Will calculate indices for: {model_names}")
    click.echo(f"Will calculate following indices: {index_names}")
