import unittest
import shutil
import os
import socket
import numpy as np
import logging
from pathlib import Path

from timewise import WiseDataByVisit, WISEDataDESYCluster, ParentSampleBase
from timewise.general import main_logger, data_dir, bigdata_dir
from timewise.utils import get_mirong_sample


main_logger.setLevel('DEBUG')
logger = logging.getLogger("timewise.test")


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


class WISEBigDataLocal(WISEDataDESYCluster):

    def __init__(self, base_name, parent_sample_class, min_sep_arcsec, n_chunks, fails=None):
        super().__init__(base_name, parent_sample_class, min_sep_arcsec, n_chunks)
        self.fails = fails

    def submit_to_cluster(
            self,
            node_memory,
            single_chunk=None,
            mask_by_position=False
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

        # make data_product files, storing essential info from parent_sample
        for jobID in range(_start_id, _end_id+1):
            indices = self.parent_sample.df.index[self.cluster_jobID_map == jobID]
            logger.debug(f"starting data_product for {len(indices)} objects.")
            data_product = self._start_data_product(parent_sample_indices=indices)
            chunk_number = self._get_chunk_number_for_job(jobID)
            self._save_data_product(data_product, service="tap", chunk_number=chunk_number, jobID=jobID)

        for job_id in range(_start_id, _end_id+1):
            logger.debug(f"Job {job_id}")
            chunk_number = self._get_chunk_number_for_job(job_id)

            if (self.fails is not None) and (job_id in self.fails):
                logger.debug(f"Job {job_id} failed")
                continue

            try:
                self._subprocess_select_and_bin(
                    service='tap',
                    chunk_number=chunk_number,
                    jobID=job_id,
                    mask_by_position=mask_by_position
                )
                self.calculate_metadata(service='tap', chunk_number=chunk_number, jobID=job_id)
            except ValueError as e:
                logger.error(f"ValueError: {e}")

        return 1

    def wait_for_job(self, job_id=None):
        logger.info("called dummy wait for cluster")

    # def clean_up(self):
    #     logger.info(f"removing {self.cache_dir}")
    #     shutil.rmtree(self.cache_dir)


class WISEBigDataTestVersion(WISEBigDataLocal):

    base_name = "test/test_mock_desy_bigdata"

    def __init__(self, base_name=base_name, fails=None):
        super().__init__(base_name=base_name,
                         parent_sample_class=MirongParentSample,
                         min_sep_arcsec=8,
                         n_chunks=2,
                         fails=fails)


####################################
# END DEFINING TEST CLASSES        #
###########################################################################################################


class TestMIRFlareCatalogue(unittest.TestCase):

    def test_a_wise_data(self):
        logger.info('\n\n Testing WISE Data \n')
        wise_data = WISEDataTestVersion()
        wise_data.match_all_chunks(additional_columns=["w1mpro"])

        logger.info('\n' + wise_data.parent_sample.df.to_string())

        wise_data.parent_sample.plot_cutout(0, arcsec=40, save=True, which="sdss")
        wise_data.parent_sample.plot_cutout(0, arcsec=40, save=True, which="panstarrs")

        logger.info(f"\n\n Testing getting photometry \n")
        for s in ['gator', 'tap']:

            logger.info(f"\nTesting {s.upper()}")
            wise_data.get_photometric_data(service=s, mask_by_position=True)

            logger.info(f" --- Test adding flux densities --- ")
            wise_data.add_flux_densities_to_saved_lightcurves(s)

            logger.info(f" --- Test adding luminosities --- ")
            wise_data.add_luminosity_to_saved_lightcurves(s, redshift_key='Z')

            logger.info(f" --- Test calculating metadata --- ")
            wise_data.calculate_metadata(service=s)

            logger.info(f" --- Test plot lightcurves --- ")
            lcs = wise_data.load_data_product(s)
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

            wise_data.plot_diagnostic_binning(service="gator", ind=int(plot_id))

    def test_b_test_match_to_wise_allsky(self):
        logger.info('\n\n Testing WISE AllSky interface \n')
        wise_data = WISEDataTestVersion(
            base_name=WISEDataTestVersion.base_name + '_match_to_allsky'
        )
        in_filename = os.path.join(wise_data.cache_dir, "test_allsky_match_in.xml")
        out_filename = os.path.join(wise_data.cache_dir, "test_allsky_match_out.tbl")
        mask = [True] * len(wise_data.parent_sample.df)
        res = wise_data._match_to_wise(
            table_name=wise_data.get_db_name("WISE All-Sky Source Catalog"),
            in_filename=in_filename,
            out_filename=out_filename,
            mask=mask,
            one_to_one=True,
        )
        logger.info(f"matched {len(res)} objects")
        self.assertEqual(len(res), len(wise_data.parent_sample.df))

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
        lcs = wise_data.load_data_product(s)
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
            wait=0,
            mask_by_position=True
        )

        logger.info(f" --- Test chi2 plots --- ")
        wise_data.make_chi2_plot(load_from_bigdata_dir=True, save=True)
        logger.info(f" --- Test coverage plots --- ")
        wise_data.make_coverage_plots(load_from_bigdata_dir=True, save=True)

        logger.info(f" --- Test plot lightcurves --- ")
        plot_id = "2"
        for lumk in ['mag', 'flux_density']:
            fn = os.path.join(wise_data.plots_dir, f"{plot_id}_{lumk}.pdf")
            wise_data.plot_lc(
                parent_sample_idx=plot_id,
                lum_key=lumk,
                service='tap',
                fn=fn,
                load_from_bigdata_dir=True
            )

    def test_d_emulate_wise_bigdata_fail(self):
        logger.info("\n\n Emulating WISEBigDataDESYCluster job fails\n\n")
        fail_job = 1
        wise_data = WISEBigDataTestVersion(fails=[fail_job])

        bigdata_phot_dir = Path(wise_data._cache_photometry_dir.replace(data_dir, bigdata_dir))
        phot_dir = Path(wise_data._cache_photometry_dir)

        for f in bigdata_phot_dir.glob("raw_photometry*"):
            if f.is_file():
                dst = phot_dir / f.name
                logger.debug(f"copying {f} back to {dst}")
                shutil.copy(f, dst)

        for f in bigdata_phot_dir.glob("timewise_data_product*"):
            if f.is_file():
                logger.debug(f"removing {f}")
                os.remove(f)

        wise_data.get_sample_photometric_data(
            max_nTAPjobs=2,
            cluster_jobs_per_chunk=2,
            query_type="positional",
            skip_input=True,
            wait=0,
            mask_by_position=True,
            skip_download=True
        )

        # verify that the failed job is not in the data product
        with self.assertRaises(KeyError):
            wise_data.load_data_product("tap", 0, fail_job, verify_contains_lightcurves=True)

        # verify that the combined chunk file has not been produced nor moved to the big data directory
        chunk_0_data_product_filename = wise_data._data_product_filename("tap", 0, use_bigdata_dir=False)
        self.assertFalse(os.path.isfile(chunk_0_data_product_filename))
        self.assertFalse(os.path.isfile(chunk_0_data_product_filename.replace(data_dir, bigdata_dir)))

        # verify that chunk 1 was processed normally
        chunk1_data_product = wise_data.load_data_product("tap", 1,
                                                          use_bigdata_dir=True,
                                                          verify_contains_lightcurves=True)
        self.assertIsInstance(chunk1_data_product, dict)

    def test_e_wise_bigdata_desy_cluster(self):
        host = socket.gethostname()
        if np.logical_or("ifh.de" in host, ("zeuthen.desy.de" in host) and ("wgs" in host)):
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
                mask_by_position=True
            )

            N_downloaded = sum([
                len(wise_desy_bigdata.load_data_product(service="tap", chunk_number=c, use_bigdata_dir=True))
                for c in range(wise_desy_bigdata.n_chunks)
            ])

            self.assertEqual(N_downloaded, len(wise_desy_bigdata.parent_sample.df))
