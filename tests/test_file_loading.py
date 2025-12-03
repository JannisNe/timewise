import pytest
from ampel.timewise.alert.load.TimewiseFileLoader import TimewiseFileLoader
from timewise.config import TimewiseConfig


def test_loader_fails(timewise_config_path, ampel_interface):
    # remove one of the files
    cfg = TimewiseConfig.from_yaml(timewise_config_path)
    dl = cfg.download.build_downloader()
    task_ids = list(dl.iter_tasks())
    task_to_remove = task_ids[0]
    dl.backend._data_path(task_to_remove).unlink()

    # make sure loader fails when file is missing
    strict_loader = TimewiseFileLoader(
        timewise_config_file=str(timewise_config_path),
        stock_id_column_name="orig_id",
        skip_missing_files=False,
    )

    # missing files should raise
    with pytest.raises(FileNotFoundError):
        list(strict_loader)

    # make sure loader skips missing files when configured to do so
    skip_loader = TimewiseFileLoader(
        timewise_config_file=str(timewise_config_path),
        stock_id_column_name="orig_id",
        skip_missing_files=True,
    )
    tables = list(skip_loader)
    assert len(tables) == sum([len(c.data) for c in dl.chunker])
