from __future__ import annotations

import json

from fax.cli import main


def test_cli_example(capsys):
    assert main(["--example"]) == 0
    captured = capsys.readouterr()
    assert "[run]" in captured.out
    assert "model = \"continuum\"" in captured.out


def test_cli_dry_run(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[run]
case = "cli"
model = "continuum"
""",
        encoding="utf-8",
    )

    assert main([str(config_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "cli"
    assert payload["status"] == "planned"


def test_cli_dump_config(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[run]\ncase = \"dump\"\n", encoding="utf-8")

    assert main([str(config_path), "--dump-config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["case"] == "dump"

