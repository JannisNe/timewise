import unittest, shutil, os, socket
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from timewise import WiseDataByVisit, WISEDataDESYCluster, ParentSampleBase
from timewise.general import main_logger
from timewise.utils import get_mirong_sample


main_logger.setLevel('DEBUG')
logger = main_logger.getChild(__name__)


mirong_sample = get_mirong_sample()
mirong_test_id = 28

test_ra = mirong_sample['RA'].iloc[mirong_test_id]
test_dec = mirong_sample['DEC'].iloc[mirong_test_id]
test_radius_arcsec = 3600


###########################################################################################################
# START DEFINING TEST CLASSES      #
####################################


class MirongParentSample(ParentSampleBase):

    default_keymap = {
        'ra': 'RA',
        'dec': 'DEC',
        'id': 'Name'
    }

    def __init__(self):
        super().__init__(base_name="test/test_mirong_parent_sample")
        self.df = get_mirong_sample()[:10]


class WISEDataTestVersion(WiseDataByVisit):
    base_name = "test/test_wise_data"

    def __init__(self, base_name=base_name):
        super().__init__(base_name=base_name,
                         parent_sample_class=MirongParentSample,
                         min_sep_arcsec=8,
                         n_chunks=2)

    def clean_up(self):
        logger.info(f"removing {self.cache_dir}")
        shutil.rmtree(self.cache_dir)


####################################
# END DEFINING TEST CLASSES        #
###########################################################################################################


class TestMIRFlareCatalogue(unittest.TestCase):

    def test_a_wise_data(self):
        logger.info('\n\n Testing WISE Data \n')
        wise_data = WISEDataTestVersion()
        wise_data.match_all_chunks()

        logger.info('\n' + wise_data.parent_sample.df.to_string())

        wise_data.parent_sample.plot_cutout(0, arcsec=40, save=True)

        logger.info(f"\n\n Testing getting photometry \n")
        for s in ['gator', 'tap']:

            logger.info(f"\nTesting {s.upper()}")
            wise_data.get_photometric_data(service=s)

            logger.info(f" --- Test adding flux densities --- ")
            wise_data.add_flux_densities_to_saved_lightcurves(s)

            logger.info(f" --- Test adding luminosities --- ")
            wise_data.add_luminosity_to_saved_lightcurves(s, redshift_key='Z')

            logger.info(f" --- Test calculating metadata --- ")
            wise_data.calculate_metadata(service=s)

            logger.info(f" --- Test plot lightcurves --- ")
            lcs = wise_data.load_binned_lcs(s)
            plot_id = list(lcs.keys())[0].split('_')[0]
            for lumk in ['mag', 'flux_density', 'luminosity']:
                fn = os.path.join(wise_data.plots_dir, f"{plot_id}_{lumk}.pdf")
                plot_unbinned = True if lumk == 'mag' else False
                wise_data.plot_lc(
                    parent_sample_idx=plot_id,
                    plot_unbinned=plot_unbinned,
                    lum_key=lumk,
                    service=s,
                    fn=fn
                )

    def test_b_test_photometry_download_by_allwise_id(self):
        logger.info('\n\n Testing WISE Data \n')
        wise_data = WISEDataTestVersion(
            base_name=WISEDataTestVersion.base_name + '_query_by_allwise_id'
        )
        wise_data.match_all_chunks()
        s = 'tap'

        logger.info(f"\nTesting {s.upper()} and query type 'by_allwise_id'")
        wise_data.get_photometric_data(
            service=s,
            query_type='by_allwise_id',
            tables=["AllWISE Multiepoch Photometry Table"]
        )

        logger.info(f" --- Test adding flux densities --- ")
        wise_data.add_flux_densities_to_saved_lightcurves(s)

        logger.info(f" --- Test adding luminosities --- ")
        wise_data.add_luminosity_to_saved_lightcurves(s, redshift_key='Z')

        logger.info(f" --- Test calculating metadata --- ")
        wise_data.calculate_metadata(service=s)

        logger.info(f" --- Test plot lightcurves --- ")
        lcs = wise_data.load_binned_lcs(s)
        plot_id = list(lcs.keys())[0].split('_')[0]
        for lumk in ['mag', 'flux_density', 'luminosity']:
            fn = os.path.join(wise_data.plots_dir, f"{plot_id}_{lumk}.pdf")
            plot_unbinned = True if lumk == 'mag' else False
            wise_data.plot_lc(
                parent_sample_idx=plot_id,
                plot_unbinned=plot_unbinned,
                lum_key=lumk,
                service=s,
                fn=fn
            )

    def test_c_wise_bigdata_desy_cluster(self):
        host = socket.gethostname()
        if np.logical_or("ifh.de" in host, "zeuthen.desy.de" in host):
            host_server = "DESY"
        else:
            host_server = None
        logger.info(f"host name is {host_server}")

        if host_server == "DESY":
            logger.info("\n\n Testing WiseBigDataDESYCLUSTER")
            wise_desy_bigdata = WISEDataDESYCluster(
                base_name="test/test_desy_bigdata",
                parent_sample_class=MirongParentSample,
                min_sep_arcsec=8,
                n_chunks=2
            )

            wise_desy_bigdata.get_sample_photometric_data(
                cluster_jobs_per_chunk=2,
                wait=0,
            )
