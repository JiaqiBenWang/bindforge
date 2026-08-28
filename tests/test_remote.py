"""Unit tests for the Slurm script renderer and config parsing (no SSH needed)."""

import os

from binderforge import remote


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("BINDFORGE_SLURM_HOST", "login01.hpc.ac.cn")
    monkeypatch.setenv("BINDFORGE_SLURM_USER", "wang")
    monkeypatch.setenv("BINDFORGE_SLURM_KEY", "/home/wang/.ssh/id_ed25519")
    monkeypatch.setenv("BINDFORGE_SLURM_PORT", "2222")
    monkeypatch.setenv("BINDFORGE_SLURM_PARTITION", "gpu")
    monkeypatch.setenv("BINDFORGE_SLURM_GRES", "gpu:1")
    monkeypatch.setenv("BINDFORGE_SLURM_CONDA", "bindforge")
    monkeypatch.setenv("BINDFORGE_SLURM_MODULES", "cuda/12,openmm")

    cfg = remote.SlurmConfig.from_env()
    assert cfg.host == "login01.hpc.ac.cn"
    assert cfg.user == "wang"
    assert cfg.key_path == "/home/wang/.ssh/id_ed25519"
    assert cfg.port == 2222
    assert cfg.partition == "gpu"
    assert cfg.gres == "gpu:1"
    assert cfg.conda_env == "bindforge"
    assert cfg.modules == "cuda/12,openmm"
    assert cfg.configured() is True


def test_config_not_configured_without_credential(monkeypatch):
    monkeypatch.delenv("BINDFORGE_SLURM_HOST", raising=False)
    monkeypatch.delenv("BINDFORGE_SLURM_USER", raising=False)
    monkeypatch.delenv("BINDFORGE_SLURM_KEY", raising=False)
    monkeypatch.delenv("BINDFORGE_SLURM_PASSWORD", raising=False)
    assert remote.SlurmConfig.from_env().configured() is False


def test_render_script_basic():
    cfg = remote.SlurmConfig(host="h", user="u", key_path="/k",
                             time="02:00:00", cpus=8, gres="gpu:1", partition="gpu")
    script = remote.render_slurm_script(
        cfg, "abc123", "target.pdb",
        {"n_designs": 4, "length": "20-25", "md_top": 2, "md_ns": 0.05,
         "dry_run": True, "seed": 7},
    )
    assert "#SBATCH --job-name=bf_abc123" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --partition=gpu" in script
    assert "bindforge" in script
    assert "--target target.pdb" in script
    assert "--n-designs 4" in script
    assert "--length 20-25" in script
    assert "--dry-run" in script
    assert "--results-dir ." in script


def test_render_script_conda_and_modules():
    cfg = remote.SlurmConfig(host="h", user="u", password="p",
                             conda_env="bindforge", modules="cuda/12, openmm")
    script = remote.render_slurm_script(cfg, "j", "t.fasta", {"dry_run": True})
    assert "source activate bindforge" in script
    assert "module load cuda/12" in script
    assert "module load openmm" in script


def test_render_script_providers_passthrough():
    cfg = remote.SlurmConfig(host="h", user="u", password="p")
    script = remote.render_slurm_script(
        cfg, "j", "t.pdb",
        {"design_provider": "boltz", "structure_provider": "chai"},
    )
    assert "--design-provider boltz" in script
    assert "--structure-provider chai" in script
