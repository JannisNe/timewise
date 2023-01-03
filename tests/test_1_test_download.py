import unittest, shutil, os, socket
import numpy as np
import logging

from timewise import WiseDataByVisit, WISEDataDESYCluster, ParentSampleBase
from timewise.general import main_logger
from timewise.utils import get_mirong_sample


main_logger.setLevel('DEBUG')
logger = logging.getLogger(__name__)


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


class WISEBigDataTestVersion(WISEDataDESYCluster):
    base_name = "test/test_mock_desy_bigdata"

    def __init__(self, base_name=base_name):
        super().__init__(base_name=base_name,
                         parent_sample_class=MirongParentSample,
                         min_sep_arcsec=8,
                         n_chunks=2)

    def submit_to_cluster(
            self,
            node_memory,
            single_chunk=None
    ):
        logger.info("emulating cluster work")

        # from timewise/wise_bigdata_desy_cluster.py
        if isinstance(single_chunk, type(None)):
            _start_id = 1
            _end_id = int(self.n_chunks*self.n_cluster_jobs_per_chunk)
        else:
            _start_id = int(single_chunk*self.n_cluster_jobs_per_chunk) + 1
            _end_id = int(_start_id + self.n_cluster_jobs_per_chunk) - 1

        logger.debug(f"Jobs from {_start_id} to {_end_id}")

        for job_id in range(_start_id, _end_id+1):
            logger.debug(f"Job {job_id}")
            chunk_number = self._get_chunk_number_for_job(job_id)
            self._subprocess_select_and_bin(service='tap', chunk_number=chunk_number, jobID=job_id)
            self.calculate_metadata(service='tap', chunk_number=chunk_number, jobID=job_id)

        return 1

    def wait_for_job(self, job_id=None):
        logger.info("called dummy wait for cluster")

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

        # wise_data.parent_sample.plot_cutout(0, arcsec=40, save=True)

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

    def test_c_emulate_wise_bigdata(self):
        logger.info("\n\n Emulating WISEBigDataDESYCluster \n\n")
        wise_data = WISEBigDataTestVersion()

        wise_data.get_sample_photometric_data(
            max_nTAPjobs=2,
            cluster_jobs_per_chunk=2,
            query_type="positional",
            skip_input=True,
            wait=0
        )

        wise_data.make_chi2_plot(load_from_bigdata_dir=True)
        wise_data.make_coverage_plots(load_from_bigdata_dir=True)

    def test_d_wise_bigdata_desy_cluster(self):
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

            N_downloaded = sum([
                len(wise_desy_bigdata._load_data_product(service="tap", chunk_number=c, use_bigdata_dir=True))
                for c in range(wise_desy_bigdata.n_chunks)
            ])

            self.assertEqual(N_downloaded, len(wise_desy_bigdata.parent_sample.df))
