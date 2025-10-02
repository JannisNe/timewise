import json
import logging
from pathlib import Path
import gzip
from typing import BinaryIO, cast

import pandas as pd
from pymongo import MongoClient
from bson import decode_file_iter

from tests.constants import DATA_DIR, V0_KEYMAP


logger = logging.getLogger(__name__)


def get_raw_reference_photometry(i) -> pd.DataFrame:
    chunk_id = i // 32

    reference_phot_files = [
        DATA_DIR / "photometry" / f"raw_photometry_{t}__chunk{chunk_id}.csv"
        for t in ["allwise_p3as_mep", "neowiser_p1bs_psd"]
    ]
    all_reference_photometry = pd.concat(
        [pd.read_csv(fn) for fn in reference_phot_files]
    ).reset_index()
    reference_mask_file = DATA_DIR / "masks" / f"position_mask_c{chunk_id}.json"
    with open(reference_mask_file, "r") as f:
        reference_bad_mask = json.load(f)
    reference_mask = all_reference_photometry.orig_id == i
    if (si := str(i)) in reference_bad_mask:
        reference_mask = reference_mask & ~all_reference_photometry.index.isin(
            reference_bad_mask[si]
        )
    reference_photometry: pd.DataFrame = all_reference_photometry.loc[
        reference_mask
    ].reset_index()

    # rename columns to v1
    for ol, nl in V0_KEYMAP:
        if ol in reference_photometry.columns:
            reference_photometry.rename(columns={ol: nl}, inplace=True)

    return reference_photometry


def get_stacked_reference_photometry(i, mode) -> None | pd.DataFrame:
    reference_path = DATA_DIR / "photometry" / f"timewise_data_product_tap_{mode}.json"
    with reference_path.open("r") as f:
        reference_data = json.load(f)

    if "timewise_lightcurve" not in reference_data[str(i)]:
        return None

    reference_lc = pd.DataFrame(reference_data[str(i)]["timewise_lightcurve"])
    reference_lc.set_index(reference_lc.index.astype(int), inplace=True)

    # rename columns to v1
    for ol, nl in V0_KEYMAP:
        if ol in reference_lc.columns:
            reference_lc.rename(columns={ol: nl}, inplace=True)

    return reference_lc


def restore_from_bson_dir(
    dump_dir: str, target_db_name: str, mongo_uri="mongodb://localhost:27017/"
):
    client = MongoClient(mongo_uri)
    db = client[target_db_name]

    dump_path = Path(dump_dir)

    num = 0
    for bson_file in dump_path.glob("*.bson.gz"):
        coll_name = bson_file.stem.replace(".bson", "")
        coll = db[coll_name]

        with gzip.open(bson_file, "rb") as f:
            f = cast(BinaryIO, f)
            docs = list(decode_file_iter(f))
            if docs:
                coll.insert_many(docs)

        logger.debug(f"Restored {coll_name} to {target_db_name}")
        num += 1

    logger.info(f"Restored {num} collections to {target_db_name}")
