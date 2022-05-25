import unittest

from timewise.utils import get_point_source_wise_data


class TestPSUtils(unittest.TestCase):

    def test_a_ps_wise_data(self):
        wd = get_point_source_wise_data(
            base_name="test/test_ps_utils",
            ra=2,
            dec=0
        )