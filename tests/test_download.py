from itertools import product
import pandas as pd
import numpy as np
import pytest
import pyvo as vo

from timewise.types import TAPJobMeta
from timewise.chunking import Chunker

from tests.dummy_tap import DummyTAPService, get_table_from_query_and_chunk


def test_chunking(download_cfg):
    chunks = Chunker(
        input_csv=download_cfg.input_csv, chunk_size=download_cfg.chunk_size
    )
    for i, chunk in enumerate(chunks):
        assert (
            len(pd.Index(chunk.data.orig_id).difference(range(i * 32, (i + 1) * 32)))
            == 0
        )


def test_downloader_fails(download_cfg):
    dl = download_cfg.build_downloader()
    dl.service = DummyTAPService(
        baseurl="", chunksize=download_cfg.chunk_size, fail_submit=True
    )
    with pytest.raises(vo.DALServiceError, match="failed submit"):
        dl.run()

    dl = download_cfg.build_downloader()
    dl.service = DummyTAPService(
        baseurl="", chunksize=download_cfg.chunk_size, fail_fetch=True
    )
    with pytest.raises(vo.DALServiceError, match="failed fetch"):
        dl.run()


def test_downloader_creates_files(download_cfg):
    dl = download_cfg.build_downloader()
    dl.service = DummyTAPService(baseurl="", chunksize=download_cfg.chunk_size)
    dl.run()

    chunks = Chunker(
        input_csv=download_cfg.input_csv, chunk_size=download_cfg.chunk_size
    )

    for q, c in product(download_cfg.queries, chunks):
        task = dl.get_task_id(c, q)
        b = dl.backend
        assert b.is_done(task)
        assert b.data_exists(task)

        assert b.meta_exists(task)
        assert TAPJobMeta(**b.load_meta(task))

        produced = dl.backend.load_data(task)
        reference = get_table_from_query_and_chunk(q.adql, c.chunk_id)

        for col in reference.colnames:
            if hasattr(reference[col], "mask"):
                m = ~(reference[col].mask & produced[col].mask)
            else:
                m = [True] * len(reference)
            assert sum(np.array(reference[col] - produced[col])[m]) == 0
