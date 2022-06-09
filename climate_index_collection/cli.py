import click


@click.command()
@click.option("--input-path", default=".", help="Path to the input data.")
@click.option("--output-path", default=".", help="Path to the output data.")
@click.option("--model-name", default="FOCI", help="Name of the input data source.")
def run(input_path, output_path, model_name):
    """Command line interface to the climate index collection."""
    click.echo(f"Will look for input data in: {input_path}")
    click.echo(f"Will write outputs to: {output_path}")
    click.echo(f"Will calculate indices for: {model_name}")
