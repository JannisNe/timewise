from pathlib import Path
import logging

from ..config import TimewiseConfig


logger = logging.getLogger(__name__)
DEFAULT_TEMPALTE_PATH = Path(__file__).parent / "template.yml"


def make_ampel_job_file(
    cfg_path: Path, mongo_db_name: str, template_path: Path | None = None
) -> Path:
    cfg = TimewiseConfig.from_yaml(cfg_path)
    template_path = template_path or DEFAULT_TEMPALTE_PATH
    logger.debug(f"loading ampel job template from {template_path}")
    with template_path.open("r") as f:
        template = f.read()

    # the timewise config makes sure there is only one original id key
    ori_id_key = cfg.download.queries[0].original_id_key

    ampel_job = (
        template.replace("TIMEWISE_CONFIG_PATH", str(cfg_path))
        .replace("ORIGINAL_ID_KEY", ori_id_key)
        .replace("MONGODB_NAME", mongo_db_name)
    )

    ampel_job_path = cfg_path.parent / f"{cfg_path.stem}_ampel_job.yml"
    logger.info(f"writing ampel job to {ampel_job_path}")
    with ampel_job_path.open("w") as f:
        f.write(ampel_job)

    return ampel_job_path
