import unittest

from timewise.general import main_logger
from timewise.point_source_utils import get_point_source_wise_data


main_logger.setLevel('DEBUG')


class TestPSUtils(unittest.TestCase):

    def test_a_ps_wise_data(self):
        wd = get_point_source_wise_data(
            base_name="test/test_ps_utils",
            ra=2,
            dec=0
        )