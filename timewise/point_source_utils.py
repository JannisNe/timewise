import logging
import pandas as pd

from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_by_visit import WiseDataByVisit


logger = logging.getLogger(__name__)


###########################################################################################################
#            START POINT SOURCE  UTILS              #
#####################################################


def get_point_source_parent_sample(base_name, ra, dec):

    class PointSourceParentSample(ParentSampleBase):
        default_keymap = {
            'ra': 'ra',
            'dec': 'dec',
            'id': 'id'
        }

        def __init__(self):

            super().__init__(base_name=base_name)

            self.base_name = base_name
            self.df = pd.DataFrame({'ra': [ra], 'dec': [dec], 'id': [base_name]})

        def save_local(self):
            logger.debug(f"not saving")

    return PointSourceParentSample


def get_point_source_wise_data(base_name, ra, dec, min_sep_arcsec=10, match=False, **kwargs):
    """
    Get a WISEData instance for a point source

    :param base_name: base name for storage in the data directory
    :type base_name: str
    :param ra: right ascencion
    :type ra: float
    :param dec: declination
    :type dec: float
    :param min_sep_arcsec: search radius in arcsec
    :type min_sep_arcsec: float
    :param match: match to AllWISE Source Catalogue
    :type match: bool
    :param kwargs: keyword arguments passed to WISEData.get_photometric_data()
    :type kwargs: dict
    :return: WISEData
    """
    ps = get_point_source_parent_sample(base_name, ra, dec)
    wd = WiseDataByVisit(n_chunks=1, base_name=base_name, parent_sample_class=ps, min_sep_arcsec=min_sep_arcsec)
    if match:
        wd.match_all_chunks()
    service = kwargs.pop('service', 'gator')
    wd.get_photometric_data(service=service, **kwargs)
    wd.plot_lc(parent_sample_idx=0, service=service)
    return wd


#####################################################
#            END POINT SOURCE  UTILS                #
###########################################################################################################
