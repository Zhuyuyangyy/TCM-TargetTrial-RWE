"""Application configuration via environment variables and defaults."""
import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "TCM-TargetTrial-RWE"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8011
    debug: bool = True
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    output_dir: Path = Path(__file__).resolve().parent.parent / "output"
    default_alpha: float = 0.05
    default_bootstrap_n: int = 1000
    default_ps_model: str = "logistic"
    max_ipw_trim: float = 10.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("TTR_HOST", "0.0.0.0"),
            port=int(os.getenv("TTR_PORT", "8011")),
            debug=os.getenv("TTR_DEBUG", "true").lower() == "true",
        )


settings = Settings.from_env()
