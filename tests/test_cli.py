from click.testing import CliRunner
from climate_index_collection.cli import run


def test_cli_run_defaults():
    """Ensure that the outputs are correct with default args."""
    runner = CliRunner()
    result = runner.invoke(run, [])
    assert result.exit_code == 0
    assert "Will look for input data in: ." in result.output
    assert "Will write outputs to: ." in result.output
    assert "Will calculate indices for: FOCI" in result.output


def test_cli_run_set_args():
    """Ensure that the outputs are correct with modified args."""
    runner = CliRunner()
    result = runner.invoke(
        run,
        [
            "--input-path",
            "inpath/",
            "--output-path",
            "outpath/",
            "--model-name",
            "CESM",
        ],
    )
    assert result.exit_code == 0
    assert "Will look for input data in: inpath/" in result.output
    assert "Will write outputs to: outpath/" in result.output
    assert "Will calculate indices for: CESM" in result.output
