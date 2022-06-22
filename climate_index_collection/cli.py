import click


from climate_index_collection.indices import ClimateIndexFunctions

all_known_indices = tuple(ClimateIndexFunctions.__members__.keys())


@click.command()
@click.option("--input-path", default=".", help="Path to the input data.")
@click.option("--output-path", default=".", help="Path to the output data.")
@click.option("--model-name", default="FOCI", help="Name of the input data source.")
@click.option(
    "--index-names", multiple=True, default=all_known_indices,
    help="Space separated list of all indices you want to calculate."
    )
def run(input_path, output_path, model_name, index_names):
    """Command line interface to the climate index collection."""
    click.echo(f"Will look for input data in: {input_path}")
    click.echo(f"Will write outputs to: {output_path}")
    click.echo(f"Will calculate indices for: {model_name}")
    click.echo(f"Will calculate following indicesr: {index_names}")
