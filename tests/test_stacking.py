import logging

from astropy.table import Table, vstack
from pymongo import MongoClient

from timewise.config import TimewiseConfig


from tests.constants import AMPEL_CONFIG_PATH, DATA_DIR


logger = logging.Logger(__name__)


def test_ingest(ampel_prepper, timewise_config_path):
    # switch out the template so only ingestion is run
    ingestion_only_template = DATA_DIR / "template_ingest_only.yml"
    ampel_prepper.template_path = ingestion_only_template
    ampel_prepper.run(timewise_config_path, AMPEL_CONFIG_PATH)

    client = MongoClient()

    # ----------------------------
    # check t0 collection
    # ----------------------------
    t0 = client["test_ampel"].get_collection("t0")
    n_in_db = t0.count_documents({})

    bp = TimewiseConfig.from_yaml(timewise_config_path).download.backend.base_path
    file_contents = vstack(
        [Table.read(f, format="fits") for f in bp.glob("download_chunk*.fits")]
    )

    missing = []
    duplicates = []
    for r in file_contents:
        f = {
            "body.ra": r["ra"],
            "body.dec": r["dec"],
            "body.mjd": r["mjd"],
        }
        n_doc = t0.count_documents(f)
        if n_doc == 0:
            missing.append(r)
        if n_doc > 1:
            duplicates.append(list(t0.find(f)))

    assert len(missing) == 0, f"Missing {len(missing)} documents in DB!\n{missing}"
    assert len(duplicates) == 0, f"Duplicate documents!\n{duplicates}"
    assert n_in_db == len(file_contents)

    # ----------------------------
    # check t1 collection
    # ----------------------------
    t1 = client["test_ampel"].get_collection("t1")
    assert t1.count_documents({}) > 0
