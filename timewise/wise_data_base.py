import os, subprocess, copy, json, tqdm, time, threading, queue, requests, abc, logging, backoff
import multiprocessing as mp
import pandas as pd
import numpy as np
import pyvo as vo
from astropy.io import ascii
import astropy.units as u
from astropy.table import Table
from astropy import constants
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import warnings

from timewise.general import main_logger, cache_dir, plots_dir, output_dir, logger_format, backoff_hndlr
from timewise.utils import StableTAPService


logger = logging.getLogger(__name__)


class WISEDataBase(abc.ABC):
    """
    Base class for WISE Data

    :param base_name: unique name to determine storage directories
    :type base_name: str
    :param parent_sample_class: class for parent sample
    :type parent_sample_class: `ParentSample` class
    :param min_sep_arcsec: minimum separation required to the parent sample sources
    :type min_sep_arcsec: float
    :param n_chunks: number of chunks in declination
    :type n_chunks: int
    """

    service_url = 'https://irsa.ipac.caltech.edu/TAP'
    service = StableTAPService(service_url)
    active_tap_phases = {"QUEUED", "EXECUTING", "RUN", "COMPLETED", "ERROR", "UNKNOWN"}
    running_tap_phases = ["QUEUED", "EXECUTING", "RUN"]
    done_tap_phases = {"COMPLETED", "ABORTED", "ERROR"}

    query_types = ['positional', 'by_allwise_id']


    table_names = pd.DataFrame([
        ('AllWISE Multiepoch Photometry Table', 'allwise_p3as_mep'),
        ('AllWISE Source Catalog', 'allwise_p3as_psd'),
        ('WISE 3-Band Cryo Single Exposure (L1b) Source Table', 'allsky_3band_p1bs_psd'),
        ('NEOWISE-R Single Exposure (L1b) Source Table', 'neowiser_p1bs_psd'),

    ], columns=['nice_table_name', 'table_name'])

    bands = ['W1', 'W2']
    flux_key_ext = "_flux"
    flux_density_key_ext = "_flux_density"
    mag_key_ext = "_mag"
    luminosity_key_ext = "_luminosity"
    error_key_ext = "_error"
    band_plot_colors = {'W1': 'r', 'W2': 'b'}

    photometry_table_keymap = {
        'AllWISE Multiepoch Photometry Table': {
            'flux': {
                'w1flux_ep':    f'W1{flux_key_ext}',
                'w1sigflux_ep': f'W1{flux_key_ext}{error_key_ext}',
                'w2flux_ep':    f'W2{flux_key_ext}',
                'w2sigflux_ep': f'W2{flux_key_ext}{error_key_ext}'
            },
            'mag': {
                'w1mpro_ep':    f'W1{mag_key_ext}',
                'w1sigmpro_ep': f'W1{mag_key_ext}{error_key_ext}',
                'w2mpro_ep':    f'W2{mag_key_ext}',
                'w2sigmpro_ep': f'W2{mag_key_ext}{error_key_ext}'
            }
        },
        'NEOWISE-R Single Exposure (L1b) Source Table': {
            'flux': {
                'w1flux':       f'W1{flux_key_ext}',
                'w1sigflux':    f'W1{flux_key_ext}{error_key_ext}',
                'w2flux':       f'W2{flux_key_ext}',
                'w2sigflux':    f'W2{flux_key_ext}{error_key_ext}'
            },
            'mag': {
                'w1mpro':       f'W1{mag_key_ext}',
                'w1sigmpro':    f'W1{mag_key_ext}{error_key_ext}',
                'w2mpro':       f'W2{mag_key_ext}',
                'w2sigmpro':    f'W2{mag_key_ext}{error_key_ext}'
            }
        }
    }

    # zero points come from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
    # published in Jarret et al. (2011): https://ui.adsabs.harvard.edu/abs/2011ApJ...735..112J/abstract
    magnitude_zeropoints = {
        'F_nu': {
            'W1': 309.54 * u.Jy,
            'W2': 171.787 * u.Jy
        },
        'Fstar_nu': {
            'W1': 306.682 * u.Jy,
            'W2': 170.663 * u.Jy
        },
        'Mag': {
            'W1': 20.752,
            'W2': 19.596
        }
    }

    aperture_corrections = {
        'W1': 0.222,
        'W2': 0.280
    }

    _this_dir = os.path.abspath(os.path.dirname(__file__))
    magnitude_zeropoints_corrections = ascii.read(f'{_this_dir}/wise_flux_conversion_correction.dat',
                                                  delimiter='\t').to_pandas()

    band_wavelengths = {
        'W1': 3.368 * 1e-6 * u.m,
        'W2': 4.618 * 1e-6 * u.m
    }

    constraints = [
        "nb < 2",
        "na < 1",
        "cc_flags like '00%'",
        "qi_fact >= 1",
        "saa_sep >= 5",
        "moon_masked like '00%'"
    ]

    parent_wise_source_id_key = 'AllWISE_id'
    parent_sample_wise_skysep_key = 'sep_to_WISE_source'

    def __init__(self,
                 base_name,
                 parent_sample_class,
                 min_sep_arcsec,
                 n_chunks):
        #######################################################################################
        # START SET-UP          #
        #########################

        self.parent_sample_class = parent_sample_class
        parent_sample = parent_sample_class() if parent_sample_class else None
        self.base_name = base_name
        self.min_sep = min_sep_arcsec * u.arcsec
        self._n_chunks = n_chunks

        # --------------------------- vvvv set up parent sample vvvv --------------------------- #
        self.parent_ra_key = parent_sample.default_keymap['ra'] if parent_sample else None
        self.parent_dec_key = parent_sample.default_keymap['dec'] if parent_sample else None
        self.parent_wise_source_id_key = WISEDataBase.parent_wise_source_id_key
        self.parent_sample_wise_skysep_key = WISEDataBase.parent_sample_wise_skysep_key
        self.parent_sample_default_entries = {
            self.parent_wise_source_id_key: "",
            self.parent_sample_wise_skysep_key: np.inf
        }

        self.parent_sample = parent_sample

        if self.parent_sample:
            for k, default in self.parent_sample_default_entries.items():
                if k not in parent_sample.df.columns:
                    self.parent_sample.df[k] = default

            self._no_allwise_source = self.parent_sample.df[self.parent_sample_wise_skysep_key] == np.inf

        else:
            self._no_allwise_source = None
        # --------------------------- ^^^^ set up parent sample ^^^^ --------------------------- #

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self._cache_photometry_dir = os.path.join(self.cache_dir, "photometry")
        self.cluster_dir = os.path.join(self.cache_dir, 'cluster')
        self.cluster_log_dir = os.path.join(self.cluster_dir, 'logs')
        self.output_dir = os.path.join(output_dir, base_name)
        self.lightcurve_dir = os.path.join(self.output_dir, "lightcurves")
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self._cache_photometry_dir, self.cluster_dir, self.cluster_log_dir,
                  self.output_dir, self.lightcurve_dir, self.plots_dir]:
            if not os.path.isdir(d):
                logger.debug(f"making directory {d}")
                os.makedirs(d)

        file_handler = logging.FileHandler(filename=self.cache_dir + '/log.err', mode="a")
        file_handler.setLevel("WARNING")
        file_handler.setFormatter(logger_format)
        logger.addHandler(file_handler)

        self.submit_file = os.path.join(self.cluster_dir, 'submit.txt')

        # set up result attributes
        self._split_chunk_key = '__chunk'
        self._cached_raw_photometry_prefix = 'raw_photometry'
        self.tap_jobs = None
        self.queue = queue.Queue()
        self.clear_unbinned_photometry_when_binning = False
        self._cached_final_products = {
            'lightcurves': dict(),
            'metadata': dict()
        }

        self._tap_wise_id_key = 'wise_id'
        self._tap_orig_id_key = 'orig_id'

        # Any class that wants to implement cluster operation has to use this variable
        # It specifies which chunks will be processed by which jobs
        self.cluster_jobID_map = None

        #########################
        # END SET-UP            #
        #######################################################################################

        #######################################################################################
        # START CHUNK MASK      #
        #########################

        self.chunk_map = None
        self.n_chunks = self._n_chunks

    @property
    def n_chunks(self):
        return self._n_chunks

    @n_chunks.setter
    def n_chunks(self, value):
        """Sets the private variable _n_chunks and re-calculates the declination interval masks"""

        if value > 50:
            logger.warning(f"Very large number of chunks ({value})! "
                           f"Pay attention when getting photometry to not kill IRSA!")

        if self.parent_sample:

            self.chunk_map = np.zeros(len(self.parent_sample.df))
            N_in_chunk = int(round(len(self.chunk_map) / self._n_chunks))
            for i in range(self._n_chunks):
                start_ind = i * N_in_chunk
                end_ind = start_ind + N_in_chunk
                self.chunk_map[start_ind:end_ind] = int(i)

            self._n_chunks = int(max(self.chunk_map)) + 1

            if self._n_chunks != value:
                logger.info(f"All objectes included in {self._n_chunks:.0f} chunks.")

        else:
            logger.warning("No parent sample given! Can not calculate dec interval masks!")

    def _get_chunk_number(self, wise_id=None, parent_sample_index=None):
        if isinstance(wise_id, type(None)) and isinstance(parent_sample_index, type(None)):
            raise Exception

        if not isinstance(wise_id, type(None)):
            parent_sample_index = np.where(self.parent_sample.df[self.parent_wise_source_id_key] == int(wise_id))[0]
            logger.debug(f"wise ID {wise_id} at index {parent_sample_index}")

        loc = self.parent_sample.df.loc[int(parent_sample_index)].name
        iloc = self.parent_sample.df.index.get_loc(loc)
        _chunk_number = int(self.chunk_map[int(iloc)])
        logger.debug(f"chunk number is {_chunk_number} for {parent_sample_index}")
        return _chunk_number

        #########################
        # END CHUNK MASK        #
        #######################################################################################

    def _start_data_product(self, parent_sample_indices):

        # get all rows in this chunk and columns, specified in the keymap
        parent_sample_sel = self.parent_sample.df.loc[
            parent_sample_indices,
            list(self.parent_sample.default_keymap.values())
        ]

        # invert the keymap to rename the columns
        inverse_keymap = {v: k for k, v in self.parent_sample.default_keymap.items()}
        parent_sample_sel.rename(columns=inverse_keymap, inplace=True)
        parent_sample_sel.set_index(parent_sample_sel.index.astype(str), inplace=True)

        # save to data_product
        data_product = parent_sample_sel.to_dict(orient="index")

        return data_product

    @staticmethod
    def get_db_name(table_name, nice=False):
        """
        Get the right table name

        :param table_name: str, table name
        :param nice: bool, whether to get the nice table name
        :return: str
        """
        source_column = 'nice_table_name' if not nice else 'table_name'
        target_column = 'table_name' if not nice else 'nice_table_name'

        m = WISEDataBase.table_names[source_column] == table_name
        if np.any(m):
            table_name = WISEDataBase.table_names[target_column][m].iloc[0]
        else:
            logger.debug(f"{table_name} not in Table. Assuming it is the right name already.")
        return table_name

    ###########################################################################################################
    # START MATCH PARENT SAMPLE TO WISE SOURCES         #
    #####################################################

    def match_all_chunks(self,
                         table_name="AllWISE Source Catalog",
                         save_when_done=True):
        """
        Some descritopn

        :param table_name: The name of the table you want to match against
        :type table_name: str
        :param save_when_done: save the parent sample dataframe with the matching info when done
        :type save_when_done: bool
        :return:
        """

        logger.info(f'matching all chunks to {table_name}')
        for i in range(self.n_chunks):
            self._match_single_chunk(i, table_name)

        _dupe_mask = self._get_dubplicated_wise_id_mask()

        self._no_allwise_source = self.parent_sample.df[self.parent_sample_wise_skysep_key] == np.inf
        if np.any(self._no_allwise_source):
            logger.warning(f"{len(self.parent_sample.df[self._no_allwise_source])} of {len(self.parent_sample.df)} "
                           f"entries without match!")

        if np.any(self._get_dubplicated_wise_id_mask()):
            logger.warning(self.parent_sample.df[self._get_dubplicated_wise_id_mask()])

        if save_when_done:
            self.parent_sample.save_local()

    def _run_gator_match(self, in_file, out_file, table_name,
                         one_to_one=True, minsep_arcsec=None, additional_keys='', silent=False, constraints=None):
        _one_to_one = '-F one_to_one=1 ' if one_to_one else ''
        _minsep_arcsec = self.min_sep.to("arcsec").value if minsep_arcsec is None else minsep_arcsec
        _db_name = self.get_db_name(table_name)
        _silent = "-s " if silent else ""
        _constraints = '-F constraints="' + " and ".join(constraints).replace('%', '%%') + '" ' if constraints else ""

        if _db_name == "allwise_p3as_mep":
            _sigpos = _source_id = _des = ""
            _id_key = "cntr_mf,cntr"
        else:
            _sigpos = 'sigra,sigdec,'
            _source_id = "source_id,"
            _des = 'designation,' if 'allwise' in _db_name else ''
            _id_key = 'cntr' if 'allwise' in _db_name else 'allwise_cntr,cntr'

        submit_cmd = f'curl ' \
                     f'--connect-timeout 3600 ' \
                     f'--max-time 3600 ' \
                     f'{_silent}' \
                     f'-o {out_file} ' \
                     f'-F filename=@{in_file} ' \
                     f'-F catalog={_db_name} ' \
                     f'-F spatial=Upload ' \
                     f'-F uradius={_minsep_arcsec} ' \
                     f'-F outfmt=1 ' \
                     f'{_one_to_one}' \
                     f'{_constraints}' \
                     f'-F selcols={_des}{_source_id}ra,dec,{_sigpos}{_id_key}{additional_keys} ' \
                     f'"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"'

        logger.debug(f'submit command: {submit_cmd}')
        N_tries = 10
        while True:
            try:
                process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
                break
            except OSError as e:
                if N_tries < 1:
                    raise OSError(e)
                logger.warning(f"{e}, retry")
                N_tries -= 1

        out_msg, err_msg = process.communicate()
        if out_msg:
            logger.info(out_msg.decode())
        if err_msg:
            logger.error(err_msg.decode())
        process.terminate()
        if os.path.isfile(out_file):
            return 1
        else:
            return 0

    def _match_to_wise(self, in_filename, out_filename, mask, table_name,
                       # remove_when_done=True,
                       N_retries=10, **gator_kwargs):
        selected_parent_sample = copy.copy(
            self.parent_sample.df.loc[mask, [self.parent_ra_key, self.parent_dec_key]])
        selected_parent_sample.rename(columns={self.parent_dec_key: 'dec',
                                               self.parent_ra_key: 'ra'},
                                      inplace=True)
        logger.debug(f"{len(selected_parent_sample)} selected")

        # write to IPAC formatted table
        _selected_parent_sample_astrotab = Table.from_pandas(selected_parent_sample, index=True)
        logger.debug(f"writing {len(_selected_parent_sample_astrotab)} "
                     f"objects to {in_filename}")
        _selected_parent_sample_astrotab.write(in_filename, format='ipac', overwrite=True)
        _done = False

        while True:
            if N_retries == 0:
                raise RuntimeError('Failed with retries')

            try:
                # use Gator to query IRSA
                success = self._run_gator_match(in_filename, out_filename, table_name, **gator_kwargs)

                if not success:
                    # if not successful try again
                    logger.warning("no success, try again")
                    continue

                # load the result file
                gator_res = Table.read(out_filename, format='ipac')
                logger.debug(f"found {len(gator_res)} results")
                return gator_res

            except ValueError:
                # this will happen if the gator match returns an output containing the error message
                # read and display error message, then try again
                with open(out_filename, 'r') as f:
                    err_msg = f.read()
                logger.warning(f"{err_msg}: try again")

            finally:
                N_retries -= 1

    def _match_single_chunk(self, chunk_number, table_name):
        """
        Match the parent sample to WISE

        :param chunk_number: number of the declination chunk
        :type chunk_number: int
        :param table_name: optional, WISE table to match to, default is AllWISE Source Catalog
        :type table_name: str
        """

        dec_intervall_mask = self.chunk_map == chunk_number
        logger.debug(f"Any selected: {np.any(dec_intervall_mask)}")
        _parent_sample_declination_band_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.xml")
        _output_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.tbl")
        gator_res = self._match_to_wise(
            in_filename=_parent_sample_declination_band_file,
            out_filename=_output_file,
            mask=dec_intervall_mask,
            table_name=table_name
        )

        for fn in [_parent_sample_declination_band_file, _output_file]:
            try:
                logger.debug(f"removing {fn}")
                os.remove(fn)
            except FileNotFoundError:
                logger.warning(f"No File!!")

        # insert the corresponding separation to the WISE source into the parent sample
        self.parent_sample.df.loc[
            dec_intervall_mask,
            self.parent_sample_wise_skysep_key
        ] = list(gator_res["dist_x"])

        # insert the corresponding WISE IDs into the parent sample
        self.parent_sample.df.loc[
            dec_intervall_mask,
            self.parent_wise_source_id_key
        ] = list(gator_res["cntr"])

        _no_match_mask = self.parent_sample.df[self.parent_sample_wise_skysep_key].isna() & dec_intervall_mask
        for k, default in self.parent_sample_default_entries.items():
            self.parent_sample.df.loc[_no_match_mask, k] = default

    def _get_dubplicated_wise_id_mask(self):
        idf_sorted_sep = self.parent_sample.df.sort_values(self.parent_sample_wise_skysep_key)
        idf_sorted_sep['duplicate'] = idf_sorted_sep[self.parent_wise_source_id_key].duplicated(keep='first')
        idf_sorted_sep.sort_index(inplace=True)
        _inf_mask = idf_sorted_sep[self.parent_sample_wise_skysep_key] < np.inf
        _dupe_mask = idf_sorted_sep['duplicate'] & (_inf_mask)
        if np.any(_dupe_mask):
            _N_dupe = len(self.parent_sample.df[_dupe_mask])
            logger.info(f"{_N_dupe} duplicated entries in parent sample")
        return _dupe_mask

    ###################################################
    # END MATCH PARENT SAMPLE TO WISE SOURCES         #
    ###########################################################################################################

    ###########################################################################################################
    # START GET PHOTOMETRY DATA       #
    ###################################

    def get_photometric_data(self, tables=None, perc=1, wait=0, service=None, nthreads=100,
                             chunks=None, overwrite=True, remove_chunks=False, query_type='positional',
                             skip_download=False):
        """
        Load photometric data from the IRSA server for the matched sample. The result will be saved under

            </path/to/timewise/data/dir>/output/<base_name>/lightcurves/binned_lightcurves_<service>.json

        :param remove_chunks: remove single chunk files after binning
        :type remove_chunks: bools
        :param overwrite: overwrite already existing lightcurves and metadata
        :type overwrite: bool
        :param tables: WISE tables to use for photometry query, defaults to AllWISE and NOEWISER photometry
        :type tables: str or list-like
        :param perc: percentage of sources to load photometry for, default 1
        :type perc: float
        :param nthreads: max number of threads to launch
        :type nthreads: int
        :param service: either of 'gator' or 'tap', selects base on elements per chunk by default
        :type service: str
        :param wait: time in hours to wait after submitting TAP jobs
        :type wait: float
        :param chunks: containing indices of chunks to download
        :type chunks: list-like
        :param query_type: 'positional': query photometry based on distance from object, 'by_allwise_id': select all photometry points within a radius of 50 arcsec with the corresponding AllWISE ID
        :type query_type: str
        :param skip_download: if `True` skip downloading and only do binning
        :type skip_download: bool
        """

        mag = True
        flux = True

        if tables is None:
            tables = [
                'AllWISE Multiepoch Photometry Table',
                'NEOWISE-R Single Exposure (L1b) Source Table'
            ]

        if query_type not in self.query_types:
            raise ValueError(f"Unknown query type {query_type}! Choose one of {self.query_types}")

        if chunks is None:
            chunks = list(range(round(int(self.n_chunks * perc))))
        else:
            cm = [c not in self.chunk_map for c in chunks]
            if np.any(cm):
                raise ValueError(f"Chunks {np.array(chunks)[cm]} are not in chunk map. "
                                 f"Probably they are larger than the set chunk number of {self._n_chunks}")

        if service is None:
            elements_per_chunk = len(self.parent_sample.df) / self.n_chunks
            service = 'tap' if elements_per_chunk > 300 else 'gator'

        if (query_type == 'by_allwise_id') and (service == 'gator'):
            raise ValueError(f"Query type 'by_allwise_id' only implemented for service 'tap'!")

        if not skip_download:

            logger.debug(f"Getting {perc * 100:.2f}% of lightcurve chunks ({len(chunks)}) via {service} "
                         f"in {'magnitude' if mag else ''} {'flux' if flux else ''} "
                         f"from {tables}")

            if service == 'tap':
                self._query_for_photometry(tables, chunks, wait, mag, flux, nthreads, query_type)

            elif service == 'gator':
                self._query_for_photometry_gator(tables, chunks, mag, flux, nthreads)

        else:
            logger.info("skipping download, assume data is already downloaded.")

        self._select_individual_lightcurves_and_bin(service=service, chunks=chunks)
        for c in chunks:
            self.calculate_metadata(service=service, chunk_number=c, overwrite=True)

        self._combine_data_products(service=service, remove=remove_chunks, overwrite=overwrite)

    def _data_product_filename(self, service, chunk_number=None, jobID=None):

        n = "timewise_data_product_"

        if (chunk_number is None) and (jobID is None):
            return os.path.join(self.lightcurve_dir, f"{n}{service}.json")
        else:
            fn = f"{n}{service}{self._split_chunk_key}{chunk_number}"
            if (chunk_number is not None) and (jobID is None):
                return os.path.join(self._cache_photometry_dir, fn + ".json")
            else:
                return os.path.join(self._cache_photometry_dir, fn + f"_{jobID}.json")

    def _load_data_product(self, service, chunk_number=None, jobID=None, return_filename=False):
        fn = self._data_product_filename(service, chunk_number, jobID)
        logger.debug(f"loading {fn}")
        try:
            with open(fn, "r") as f:
                lcs = json.load(f)
            if return_filename:
                return lcs, fn
            return lcs
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_data_product(self, data_product, service, chunk_number=None, jobID=None, overwrite=False):
        fn = self._data_product_filename(service, chunk_number, jobID)
        logger.debug(f"saving {len(data_product)} new lightcurves to {fn}")

        if fn == self._data_product_filename(service):
            self._cached_final_products['lightcurves'][service] = data_product

        if not overwrite:
            try:
                old_data_product = self._load_data_product(service=service, chunk_number=chunk_number, jobID=jobID)
                logger.debug(f"Found {len(old_data_product)}. Combining")
                data_product = data_product.update(old_data_product)
            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new binned lightcurves.")

        with open(fn, "w") as f:
            json.dump(data_product, f, indent=4)

    def _lightcurve_filename(self, service, chunk_number=None, jobID=None):

        warnings.warn("Separate `binned_lightcurves` and `metadata` will be deprecated in v0.3.0!", DeprecationWarning)

        if (chunk_number is None) and (jobID is None):
            return os.path.join(self.lightcurve_dir, f"binned_lightcurves_{service}.json")
        else:
            fn = f"binned_lightcurves_{service}{self._split_chunk_key}{chunk_number}"
            if (chunk_number is not None) and (jobID is None):
                return os.path.join(self._cache_photometry_dir, fn + ".json")
            else:
                return os.path.join(self._cache_photometry_dir, fn + f"_{jobID}.json")

    def _load_lightcurves(self, service, chunk_number=None, jobID=None, return_filename=False):
        fn = self._lightcurve_filename(service, chunk_number, jobID)
        logger.debug(f"loading {fn}")
        try:
            with open(fn, "r") as f:
                lcs = json.load(f)
            if return_filename:
                return lcs, fn
            return lcs
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_lightcurves(self, lcs, service, chunk_number=None, jobID=None, overwrite=False):
        fn = self._lightcurve_filename(service, chunk_number, jobID)
        logger.debug(f"saving {len(lcs)} new lightcurves to {fn}")

        if fn == self._lightcurve_filename(service):
            self._cached_final_products['lightcurves'][service] = lcs

        if not overwrite:
            try:
                old_lcs = self._load_lightcurves(service=service, chunk_number=chunk_number, jobID=jobID)
                logger.debug(f"Found {len(old_lcs)}. Combining")
                lcs = lcs.update(old_lcs)
            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new binned lightcurves.")

        with open(fn, "w") as f:
            json.dump(lcs, f)

    def load_binned_lcs(self, service):
        """Loads the binned lightcurves. For any int `ID` the lightcurves can convieniently read into a pandas.DataFrame
        via

            lc = pandas.DataFrame.from_dict(json_dictionary[ID])

        :param service: the service with which the lightcuvres were downloaded
        :type service: str
        :return: the binned lightcurves
        :rtype: dict
        """
        if service not in self._cached_final_products['lightcurves']:
            self._cached_final_products['lightcurves'][service] = self._load_data_product(service)
        return self._cached_final_products['lightcurves'][service]

    def _combine_data_products(self, service=None, chunk_number=None, remove=False, overwrite=False):
        if not service:
            logger.info("Combining all lightcuves collected with all services")
            itr = ['service', ['gator', 'tap']]
            kwargs = {}
        elif chunk_number is None:
            logger.info(f"Combining all lightcurves collected with {service}")
            itr = ['chunk_number', range(self.n_chunks)]
            kwargs = {'service': service}
        elif chunk_number is not None:
            logger.info(f"Combining all lightcurves collected eith {service} for chunk {chunk_number}")
            itr = ['jobID',
                   list(self.clusterJob_chunk_map.index[self.clusterJob_chunk_map.chunk_number == chunk_number])]
            kwargs = {'service': service, 'chunk_number': chunk_number}
        else:
            raise NotImplementedError

        lcs = None
        fns = list()
        for i in itr[1]:
            kw = dict(kwargs)
            kw[itr[0]] = i
            kw['return_filename'] = True
            res = self._load_data_product(**kw)
            if not isinstance(res, type(None)):
                ilcs, ifn = res
                fns.append(ifn)
                if isinstance(lcs, type(None)):
                    lcs = dict(ilcs)
                else:
                    lcs.update(ilcs)

        self._save_data_product(lcs, service=service, chunk_number=chunk_number, overwrite=overwrite)

        if remove:
            for fn in tqdm.tqdm(fns, desc="removing files"):
                os.remove(fn)

    # ----------------------------------------------------------------------------------- #
    # START using GATOR to get photometry        #
    # ------------------------------------------ #

    def _gator_chunk_photometry_cache_filename(self, table_nice_name, chunk_number,
                                               additional_neowise_query=False, gator_input=False):
        table_name = self.get_db_name(table_nice_name)
        _additional_neowise_query = '_neowise_gator' if additional_neowise_query else ''
        _gator_input = '_gator_input' if gator_input else ''
        _ending = '.xml' if gator_input else'.tbl'
        fn = f"{self._cached_raw_photometry_prefix}_{table_name}{_additional_neowise_query}{_gator_input}" \
             f"{self._split_chunk_key}{chunk_number}{_ending}"
        return os.path.join(self._cache_photometry_dir, fn)

    def _thread_query_photometry_gator(self, chunk_number, table_name, mag, flux):
        _infile = self._gator_chunk_photometry_cache_filename(table_name, chunk_number, gator_input=True)
        _outfile = self._gator_chunk_photometry_cache_filename(table_name, chunk_number)
        _nice_name = self.get_db_name(table_name, nice=True)
        _additional_keys_list = ['mjd']
        if mag:
            _additional_keys_list += list(self.photometry_table_keymap[_nice_name]['mag'].keys())
        if flux:
            _additional_keys_list += list(self.photometry_table_keymap[_nice_name]['flux'].keys())

        _additional_keys = "," + ",".join(_additional_keys_list)
        _deci_mask = self.chunk_map == chunk_number
        _mask = _deci_mask #& (~self._no_allwise_source)

        res = self._match_to_wise(
            in_filename=_infile,
            out_filename=_outfile,
            mask=_mask,
            table_name=table_name,
            one_to_one=False,
            additional_keys=_additional_keys,
            minsep_arcsec=self.min_sep.to('arcsec').value,
            silent=True,
            constraints=self.constraints
        )

        os.remove(_infile)
        return res

    def _gator_photometry_worker_thread(self):
        while True:
            try:
                args = self.queue.get(block=False)
            except (AttributeError, queue.Empty):
                logger.debug('No more tasks, exiting')
                break
            logger.debug(f"{args}")
            self._thread_query_photometry_gator(*args)
            self.queue.task_done()
            logger.info(f"{self.queue.qsize()} tasks remaining")

    def _query_for_photometry_gator(self, tables, chunks, mag, flux, nthreads):
        nthreads = min(nthreads, len(chunks))
        logger.debug(f'starting {nthreads} workers')
        threads = [threading.Thread(target=self._gator_photometry_worker_thread) for _ in range(nthreads)]

        logger.debug(f"using {len(chunks)} chunks")
        self.queue = queue.Queue()
        for t in np.atleast_1d(tables):
            for i in chunks:
                self.queue.put([i, t, mag, flux])

        logger.info(f"added {self.queue.qsize()} tasks to queue")
        for t in threads:
            t.start()
        self.queue.join()
        self.queue = None

        for t in threads:
            t.join()

    def _get_unbinned_lightcurves_gator(self, chunk_number, clear=False):
        # load only the files for this chunk
        fns = [os.path.join(self._cache_photometry_dir, fn)
               for fn in os.listdir(self._cache_photometry_dir)
               if (fn.startswith(self._cached_raw_photometry_prefix) and
                   fn.endswith(f"{self._split_chunk_key}{chunk_number}.tbl"))
               ]

        logger.debug(f"chunk {chunk_number}: loading {len(fns)} files for chunk {chunk_number}")

        _data = list()
        for fn in fns:
            data_table = Table.read(fn, format='ipac').to_pandas()

            t = 'allwise_p3as_mep' if 'allwise' in fn else 'neowiser_p1bs_psd'
            nice_name = self.get_db_name(t, nice=True)
            cols = {'index_01': self._tap_orig_id_key}
            cols.update(self.photometry_table_keymap[nice_name]['mag'])
            cols.update(self.photometry_table_keymap[nice_name]['flux'])
            if 'allwise' in fn:
                cols['cntr_mf'] = 'allwise_cntr'

            data_table = data_table.rename(columns=cols)
            _data.append(data_table)

            if clear:
                os.remove(fn)

        lightcurves = pd.concat(_data)
        return lightcurves

    # ------------------------------------------ #
    # END using GATOR to get photometry          #
    # ----------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------------------- #
    # START using TAP to get photometry        #
    # ---------------------------------------- #

    def _get_photometry_query_string(self, table_name, mag, flux, query_type):
        """
        Construct a query string to submit to IRSA
        :param table_name: str, table name
        :type table_name: str
        :return: str
        """
        logger.debug(f"constructing query for {table_name}")
        db_name = self.get_db_name(table_name)
        nice_name = self.get_db_name(table_name, nice=True)
        id_key = 'cntr_mf' if 'allwise' in db_name else 'allwise_cntr'
        lum_keys = list()
        if mag:
            lum_keys += list(self.photometry_table_keymap[nice_name]['mag'].keys())
        if flux:
            lum_keys += list(self.photometry_table_keymap[nice_name]['flux'].keys())
        keys = ['ra', 'dec', 'mjd', id_key] + lum_keys
        _constraints = list(self.constraints)

        q = 'SELECT \n\t'
        for k in keys:
            q += f'{db_name}.{k}, '
        q += f'\n\tmine.{self._tap_orig_id_key} \n'
        q += f'FROM\n\tTAP_UPLOAD.ids AS mine \n'

        if query_type == 'positional':
            q += f'RIGHT JOIN\n\t{db_name} \n'
            radius = self.min_sep

        if query_type == 'by_allwise_id':
            q += f'INNER JOIN\n\t{db_name} ON {db_name}.{id_key} = mine.{self._tap_wise_id_key} \n'
            radius = 15 * u.arcsec

        q += 'WHERE \n'

        if query_type == 'positional':
            q += f"\tCONTAINS(POINT('J2000',{db_name}.ra,{db_name}.dec)," \
                 f"CIRCLE('J2000',mine.ra_in,mine.dec_in,{radius.to('deg').value}))=1 "

        if len(_constraints) > 0:

            if query_type == 'positional':
                q += ' AND (\n'

            for c in _constraints:
                q += f'\t{db_name}.{c} AND \n'
            q = q.strip(" AND \n")

            if query_type == 'positional':
                q += '\t)'

        logger.debug(f"\n{q}")
        return q

    def _submit_job_to_TAP(self, chunk_number, table_name, mag, flux, query_type):
        i = chunk_number
        t = table_name
        m = self.chunk_map == i

        # if perc is smaller than one select only a subset of wise IDs
        sel = self.parent_sample.df[np.array(m)]

        tab_d = dict()

        tab_d[self._tap_orig_id_key] = np.array(sel.index).astype(int)
        tab_d['ra_in'] = np.array(sel[self.parent_sample.default_keymap['ra']]).astype(float)
        tab_d['dec_in'] = np.array(sel[self.parent_sample.default_keymap['dec']]).astype(float)

        if query_type == 'by_allwise_id':
            tab_d[self._tap_wise_id_key] = np.array(sel[self.parent_wise_source_id_key]).astype(int)

        del sel

        logger.debug(f"{chunk_number}th query of {table_name}: uploading {len(list(tab_d.values())[0])} objects.")
        qstring = self._get_photometry_query_string(t, mag, flux, query_type)

        N_tries = 5
        while True:
            if N_tries == 0:
                logger.warning("No more tries left!")
                raise vo.dal.exceptions.DALServiceError(f"Submission failed "
                                                        f"for {i}th chunk "
                                                        f"of {t} "
                                                        f"after {N_tries} attempts")
            try:
                job = WISEDataBase.service.submit_job(qstring, uploads={'ids': Table(tab_d)})
                job.run()

                if isinstance(job.phase, type(None)):
                    raise vo.dal.DALServiceError(f"Job submission failed. No phase!")

                logger.info(f'submitted job for {t} for chunk {i}: ')
                logger.debug(f'Job: {job.url}; {job.phase}')
                self.tap_jobs[t][i] = job
                self.queue.put((t, i))
                break

            except (requests.exceptions.ConnectionError, vo.dal.exceptions.DALServiceError) as e:
                wait = 60
                N_tries -= 1
                logger.warning(f"{chunk_number}th query of {table_name}: Could not submit TAP job!\n"
                               f"{e}. Waiting {wait}s and try again. {N_tries} tries left.")
                time.sleep(wait)

    def _chunk_photometry_cache_filename(self, table_nice_name, chunk_number, additional_neowise_query=False):
        table_name = self.get_db_name(table_nice_name)
        _additional_neowise_query = '_neowise_gator' if additional_neowise_query else ''
        fn = f"{self._cached_raw_photometry_prefix}_{table_name}{_additional_neowise_query}" \
             f"{self._split_chunk_key}{chunk_number}.csv"
        return os.path.join(self._cache_photometry_dir, fn)

    @staticmethod
    def _give_up_tap(e):
        return ("Job is not active!" in str(e))

    @backoff.on_exception(
        backoff.expo,
        vo.dal.exceptions.DALServiceError,
        giveup=_give_up_tap,
        max_tries=50,
        on_backoff=backoff_hndlr
    )
    def _thread_wait_and_get_results(self, t, i):
        logger.info(f"Waiting on {i}th query of {t} ........")

        _job = self.tap_jobs[t][i]
        _job.wait()
        logger.info(f'{i}th query of {t}: Done!')

        lightcurve = _job.fetch_result().to_table().to_pandas()
        fn = self._chunk_photometry_cache_filename(t, i)
        logger.debug(f"{i}th query of {t}: saving under {fn}")

        table_nice_name = self.get_db_name(t, nice=True)
        cols = dict(self.photometry_table_keymap[table_nice_name]['mag'])
        cols.update(self.photometry_table_keymap[table_nice_name]['flux'])

        if 'allwise' in t:
            cols['cntr_mf'] = 'allwise_cntr'

        lightcurve.rename(columns=cols).to_csv(fn)
        return

    def _tap_photometry_worker_thread(self):
        while True:
            try:
                t, i = self.queue.get(block=False)
            except queue.Empty:
                logger.debug("No more tasks, exiting")
                break
            except AttributeError:
                logger.debug(f"No more queue. exiting")
                break

            job = self.tap_jobs[t][i]

            _ntries = 10
            while True:
                try:
                    job._update(timeout=600)
                    phase = job._job.phase
                    break
                except vo.dal.exceptions.DALServiceError as e:
                    msg = f"{i}th query of {t}: DALServiceError: {e}; trying again in 6 min"
                    if _ntries < 10:
                        msg += f' ({_ntries} tries left)'

                    logger.warning(msg)
                    time.sleep(60 * 6)
                    if '404 Client Error: Not Found for url' in str(e):
                        _ntries -= 1

            if phase in self.running_tap_phases:
                self.queue.put((t, i))
                self.queue.task_done()

            elif phase in self.done_tap_phases:
                self._thread_wait_and_get_results(t, i)
                self.queue.task_done()
                logger.info(f'{self.queue.qsize()} tasks left')

            else:
                logger.warning(f'queue {i} of {t}: Job not active! Phase is {phase}')

            time.sleep(np.random.uniform(60))

        logger.debug("closing thread")

    def _run_tap_worker_threads(self, nthreads):
        threads = [threading.Thread(target=self._tap_photometry_worker_thread)
                   for _ in range(nthreads)]

        for t in threads:
            t.start()

        try:
            self.queue.join()
        except KeyboardInterrupt:
            pass

        logger.info('all tap_jobs done!')
        for i, t in enumerate(threads):
            logger.debug(f"{i}th thread alive: {t.is_alive()}")

        for t in threads:
            t.join()
        self.tap_jobs = None
        del threads

    def _query_for_photometry(self, tables, chunks, wait, mag, flux, nthreads, query_type):
        # ----------------------------------------------------------------------
        #     Do the query
        # ----------------------------------------------------------------------
        self.tap_jobs = dict()
        self.queue = queue.Queue()
        tables = np.atleast_1d(tables)

        for t in tables:
            self.tap_jobs[t] = dict()
            for i in chunks:
                self._submit_job_to_TAP(i, t, mag, flux, query_type)
                time.sleep(5)

        logger.info(f'added {self.queue.qsize()} tasks to queue')
        logger.info(f"wait for {wait} hours to give tap_jobs some time")
        time.sleep(wait * 3600)
        nthreads = min(len(tables) * len(chunks), nthreads)
        self._run_tap_worker_threads(nthreads)
        self.queue = None

    # ----------------------------------------------------------------------
    #     select individual lightcurves and bin
    # ----------------------------------------------------------------------

    def _select_individual_lightcurves_and_bin(self, ncpu=35, service='tap', chunks=None):
        logger.info('selecting individual lightcurves and bin ...')
        ncpu = min(self.n_chunks, ncpu)
        logger.debug(f"using {ncpu} CPUs")
        chunk_list = list(range(self.n_chunks)) if not chunks else chunks
        service_list = [service] * len(chunk_list)
        logger.debug(f"multiprocessing arguments: chunks: {chunk_list}, service: {service_list}")

        while True:
            try:
                logger.debug(f'trying with {ncpu}')
                p = mp.Pool(ncpu)
                break
            except OSError as e:
                logger.warning(e)
                if ncpu == 1:
                    break
                ncpu = int(round(ncpu - 1))

        if ncpu > 1:
            r = list(
                tqdm.tqdm(
                    p.starmap(
                        self._subprocess_select_and_bin,
                        zip(service_list, chunk_list)
                    ),
                    total=self.n_chunks,
                    desc='select and bin'
                )
            )
            p.close()
            p.join()
        else:
            r = list(map(self._subprocess_select_and_bin, service_list, chunk_list))

    def _get_unbinned_lightcurves(self, chunk_number, clear=False):
        # load only the files for this chunk
        fns = [os.path.join(self._cache_photometry_dir, fn)
               for fn in os.listdir(self._cache_photometry_dir)
               if (fn.startswith(self._cached_raw_photometry_prefix) and fn.endswith(
                f"{self._split_chunk_key}{chunk_number}.csv"
            ))]
        logger.debug(f"chunk {chunk_number}: loading {len(fns)} files for chunk {chunk_number}")
        lightcurves = pd.concat([pd.read_csv(fn) for fn in fns])

        if clear:
            for fn in fns:
                os.remove(fn)

        return lightcurves

    def _subprocess_select_and_bin(self, service, chunk_number=None, jobID=None):
        # run through the ids and bin the lightcurves
        if service == 'tap':
            lightcurves = self._get_unbinned_lightcurves(
                chunk_number,
                clear=self.clear_unbinned_photometry_when_binning
            )
        elif service == 'gator':
            lightcurves = self._get_unbinned_lightcurves_gator(
                chunk_number,
                clear=self.clear_unbinned_photometry_when_binning
            )
        else:
            raise ValueError(f"Service {service} not known!")

        if jobID:
            indices = np.where(self.cluster_jobID_map == jobID)[0]
        else:
            indices = lightcurves[self._tap_orig_id_key].unique()

        logger.debug(f"chunk {chunk_number}: going through {len(indices)} IDs")

        data_product = self._load_data_product(service=service, chunk_number=chunk_number, jobID=jobID)

        if data_product is None:
            logger.info(f"Starting data product for {len(indices)} indices.")
            data_product = self._start_data_product(parent_sample_indices=indices)

        for parent_sample_entry_id in tqdm.tqdm(indices):
            m = lightcurves[self._tap_orig_id_key] == parent_sample_entry_id
            lightcurve = lightcurves[m]

            if len(lightcurve) < 1:
                logger.warning(f"No data for {parent_sample_entry_id}")
                continue

            binned_lc = self.bin_lightcurve(lightcurve)
            data_product[str(int(parent_sample_entry_id))]["timewise_lightcurve"] = binned_lc.to_dict()

        logger.debug(f"chunk {chunk_number}: saving {len(data_product.keys())} binned lcs")
        self._save_data_product(data_product, service=service, chunk_number=chunk_number, jobID=jobID, overwrite=True)

    # ---------------------------------------- #
    # END using TAP to get photometry          #
    # ----------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    #     bin lightcurves
    # ----------------------------------------------------------------------

    @abc.abstractmethod
    def bin_lightcurve(self, lightcurve):
        """
        Bins a lightcurve

        :param lightcurve: The unbinned lightcurve
        :type lightcurve: pandas.DataFrame
        :return: the binned lightcurve
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    #     bin lightcurves
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------------------- #
    # START converting to flux densities                   #
    # ---------------------------------------------------- #

    def find_color_correction(self, w1_minus_w2):
        """
        Find the color correction based on the W1-W2 color.
        See `this <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux>`_

        :param w1_minus_w2:
        :type w1_minus_w2: float
        :return: the color correction factor
        :rtype: float
        """
        w1_minus_w2 = np.atleast_1d(w1_minus_w2)
        c = pd.DataFrame(columns=self.magnitude_zeropoints_corrections.columns)
        power_law_values = self.magnitude_zeropoints_corrections.loc[8:16]['[W1 - W2]']
        for w1mw2 in w1_minus_w2:
            dif = power_law_values - w1mw2
            i = abs(dif).argmin()
            c = c.append(self.magnitude_zeropoints_corrections.loc[i])
        return c

    def vegamag_to_flux_density(self, vegamag, band, unit='mJy', color_correction=None):
        """
        This converts the detector level brightness m in Mag_vega to a flux density F

                    F = (F_nu / f_c) * 10 ^ (-m / 2.5)

        where F_nu is the zeropoint flux for the corresponding band and f_c a color correction factor.
        See `this <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux>`_

        :param vegamag:
        :type vegamag: float or numpy.ndarray
        :param band:
        :type band: str
        :param unit: unit to convert the flux density to
        :type unit: str
        :param color_correction: the colorcorection factor, if dict the keys have to be 'f_c("band")'
        :type color_correction: float or numpy.ndarray or dict
        :return: the flux densities
        :rtype: ndarray
        """
        if not isinstance(color_correction, type(None)):
            key = f'f_c({band})'
            if key in color_correction:
                color_correction = color_correction[key]
                if len(color_correction) != len(vegamag):
                    raise ValueError(f"\nLength of color corrections: {len(color_correction)}:\n{color_correction}; "
                                     f"\nLentgh of mags: {len(vegamag)}: \n{vegamag}")
            else:
                raise NotImplementedError(color_correction)

        else:
            color_correction = 1

        color_correction = np.array(color_correction)
        vegamag = np.array(vegamag)
        fd = self.magnitude_zeropoints['F_nu'][band].to(unit).value / color_correction * 10 ** (-vegamag / 2.5)
        if len(fd) != len(vegamag):
            raise ValueError(f"\nLength of flux densities: {len(fd)}:\n{fd}; "
                             f"\nLentgh of mags: {len(vegamag)}: \n{vegamag}")

        return np.array(list(fd))

    def add_flux_density(self, lightcurve,
                         mag_key, emag_key, mag_ul_key,
                         f_key, ef_key, f_ul_key, do_color_correction=False):
        """Adds flux densities to a lightcurves

        :param lightcurve:
        :type lightcurve: pandas.DataFrame
        :param mag_key: the key in `lightcurve` that holds the magnitude
        :type mag_key: str
        :param emag_key: the key in `lightcurve` that holds the error of the magnitude
        :type emag_key: str
        :param mag_ul_key: the key in `lightcurve` that holds the upper limit for the magnitude
        :type mag_ul_key: str
        :param f_key: the key that will hold the flux density
        :type f_key: str
        :param ef_key: the key that will hold the flux density error
        :type ef_key: str
        :param f_ul_key: the key that will hold the flux density upper limit
        :type f_ul_key: str
        :param do_color_correction:
        :type do_color_correction: bool
        :return: the lightcurve with flux density
        :rtype: pandas.DataFrame
        """

        if isinstance(lightcurve, dict):
            lightcurve = pd.DataFrame.from_dict(lightcurve, orient='columns')

        if do_color_correction:
            w1_minus_w2 = lightcurve[f"W1{mag_key}"] - lightcurve[f"W2{mag_key}"]
            f_c = self.find_color_correction(w1_minus_w2)
        else:
            f_c = None

        for b in self.bands:
            mags = lightcurve[f'{b}{mag_key}']
            emags = lightcurve[f'{b}{emag_key}']

            flux_densities = self.vegamag_to_flux_density(mags, band=b)
            upper_eflux_densities = self.vegamag_to_flux_density(mags - emags, band=b, color_correction=f_c)
            lower_eflux_densities = self.vegamag_to_flux_density(mags + emags, band=b, color_correction=f_c)
            eflux_densities = upper_eflux_densities - lower_eflux_densities

            lightcurve[f'{b}{f_key}']  = flux_densities
            lightcurve[f'{b}{ef_key}'] = eflux_densities
            if mag_ul_key:
                lightcurve[f'{b}{f_ul_key}'] = lightcurve[f'{b}{mag_ul_key}']

        return lightcurve

    def add_flux_densities_to_saved_lightcurves(self, service):
        """Adds flux densities to all downloaded lightcurves

        :param service: The service with which the lightcurves were downloaded
        :type service: str
        """
        data_product = self.load_binned_lcs(service=service)
        for i, i_data_product in tqdm.tqdm(data_product.items(), desc='adding flux densities'):
            data_product[i]["timewise_lightcurve"] = self.add_flux_density(
                i_data_product["timewise_lightcurve"],
                mag_key=f'{self.mean_key}{self.mag_key_ext}',
                emag_key=f'{self.mag_key_ext}{self.rms_key}',
                mag_ul_key=f'{self.mag_key_ext}{self.upper_limit_key}',
                f_key=f'{self.mean_key}{self.flux_density_key_ext}',
                ef_key=f'{self.flux_density_key_ext}{self.rms_key}',
                f_ul_key=f'{self.flux_density_key_ext}{self.upper_limit_key}'
            ).to_dict()
        self._save_data_product(data_product, service=service, overwrite=True)

    # ---------------------------------------------------- #
    # END converting to flux densities                     #
    # ----------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------------------- #
    # START converting to luminosity                       #
    # ---------------------------------------------------- #

    def luminosity_from_flux_density(self, flux_density, band, distance=None, redshift=None,
                                     unit='erg s-1', flux_density_unit='mJy'):
        """
        Converts a flux density into a luminosity

        :param flux_density:
        :type flux_density: float or numpy.ndarray
        :param band:
        :type band: str
        :param distance: distance to source, if not given will use luminosity distance from redshift
        :type distance: astropy.Quantity
        :param redshift: redshift to use when calculating luminosity distance
        :type redshift: float
        :param unit: unit in which to give the luminosity, default is erg s-1 sm-2
        :type unit: str or astropy.unit
        :param flux_density_unit: unit in which the flux density is given, default is mJy
        :type flux_density_unit: str or astropy.unit
        :return: the resulting luminosities
        :rtype: float or ndarray
        """

        if not distance:
            if not redshift:
                raise ValueError('Either redshift or distance has to be given!')
            else:
                distance = Planck18.luminosity_distance(float(redshift))

        F_nu = np.array(flux_density) * u.Unit(flux_density_unit) * 4 * np.pi * distance ** 2
        nu = constants.c / self.band_wavelengths[band]
        luminosity = F_nu * nu
        return luminosity.to(unit).value

    def _add_luminosity(self, lightcurve, f_key, ef_key, f_ul_key, lum_key, elum_key, lum_ul_key, **lum_kwargs):
        for band in self.bands:
            fd = lightcurve[band + f_key]
            fd_e = lightcurve[band + ef_key]
            l = self.luminosity_from_flux_density(fd, band, **lum_kwargs)
            el = self.luminosity_from_flux_density(fd_e, band, **lum_kwargs)
            lightcurve[band + lum_key] = l
            lightcurve[band + elum_key] = el
            lightcurve[band + lum_ul_key] = lightcurve[band + f_ul_key]
        return lightcurve

    def add_luminosity_to_saved_lightcurves(self, service, redshift_key=None, distance_key=None):
        """Add luminosities to all lightcurves, calculated from flux densities and distance or redshift

        :param service: the service with which the lightcurves were downloaded
        :type service: str
        :param redshift_key: the key in the parent sample data frame that holds the redshift info
        :type redshift_key: str
        :param distance_key: the key in the parent sample data frame that holds the distance info
        :type distance_key: str
        """

        if (not redshift_key) and (not distance_key):
            raise ValueError('Either distance key or redshift key has to be given!')

        data_product = self.load_binned_lcs(service=service)
        for i, i_data_product in tqdm.tqdm(data_product.items(), desc='adding luminosities'):
            parent_sample_idx = int(i.split('_')[0])
            info = self.parent_sample.df.loc[parent_sample_idx]

            if distance_key:
                distance = info[distance_key]
                redshift = None
            else:
                distance = None
                redshift = info[redshift_key]

            data_product[i]["timewise_lightcurve"] = self._add_luminosity(
                pd.DataFrame.from_dict(i_data_product["timewise_lightcurve"]),
                f_key     = self.mean_key + self.flux_density_key_ext,
                ef_key    = self.flux_density_key_ext + self.rms_key,
                f_ul_key  = self.flux_density_key_ext + self.upper_limit_key,
                lum_key   = self.mean_key + self.luminosity_key_ext,
                elum_key  = self.luminosity_key_ext + self.rms_key,
                lum_ul_key= self.luminosity_key_ext + self.upper_limit_key,
                redshift  = redshift,
                distance  = distance
            ).to_dict()
        self._save_data_product(data_product, service=service, overwrite=True)

    # ---------------------------------------------------- #
    # END converting to luminosity                         #
    # ----------------------------------------------------------------------------------- #

    #################################
    # END GET PHOTOMETRY DATA       #
    ###########################################################################################################

    ###########################################################################################################
    # START MAKE PLOTTING FUNCTIONS     #
    #####################################

    def plot_lc(self, parent_sample_idx, service='tap', plot_unbinned=False, plot_binned=True,
                interactive=False, fn=None, ax=None, save=True, lum_key='flux_density', **kwargs):
        """Make a pretty plot of a lightcurve

        :param parent_sample_idx: The index in the parent sample of the lightcurve
        :type parent_sample_idx: int
        :param service: the service with which the lightcurves were downloaded
        :type service: str
        :param plot_unbinned: plot unbinned data
        :type plot_unbinned: bool
        :param plot_binned: plot binned lightcurve
        :type plot_binned: bool
        :param interactive: interactive mode
        :type interactive: bool
        :param fn: filename, defaults to </path/to/timewise/data/dir>/output/plots/<base_name>/<parent_sample_index>_<lum_key>.pdf
        :type fn: str
        :param ax: pre-existing matplotlib.Axis
        :param save: save the plot
        :type save: bool
        :param lum_key: the unit of luminosity to use in the plot, either of 'mag', 'flux_density' or 'luminosity'
        :param kwargs: any additional kwargs will be passed on to `matplotlib.pyplot.subplots()`
        :return: the matplotlib.Figure and matplotlib.Axes if `interactive=True`
        """

        logger.debug(f"loading binned lightcurves")
        data_product = self.load_binned_lcs(service)
        _get_unbinned_lcs_fct = self._get_unbinned_lightcurves if service == 'tap' else self._get_unbinned_lightcurves_gator

        wise_id = self.parent_sample.df.loc[int(parent_sample_idx), self.parent_wise_source_id_key]
        if isinstance(wise_id, float) and not np.isnan(wise_id):
            wise_id = int(wise_id)
        logger.debug(f"{wise_id} for {parent_sample_idx}")

        lc = pd.DataFrame.from_dict(data_product[str(int(parent_sample_idx))]["timewise_lightcurve"])

        if plot_unbinned:
            _chunk_number = self._get_chunk_number(parent_sample_index=parent_sample_idx)

            if service == 'tap':
                unbinned_lcs = self._get_unbinned_lightcurves(_chunk_number)

            else:
                unbinned_lcs = self._get_unbinned_lightcurves_gator(_chunk_number)

            unbinned_lc = unbinned_lcs[unbinned_lcs[self._tap_orig_id_key] == int(parent_sample_idx)]

        else:
            unbinned_lc = None

        _lc = lc if plot_binned else None

        if not fn:
            fn = os.path.join(self.plots_dir, f"{parent_sample_idx}_{lum_key}.pdf")

        return self._plot_lc(lightcurve=_lc, unbinned_lc=unbinned_lc, interactive=interactive, fn=fn, ax=ax,
                             save=save, lum_key=lum_key, **kwargs)

    def _plot_lc(self, lightcurve=None, unbinned_lc=None, interactive=False, fn=None, ax=None, save=True,
                 lum_key='flux_density', **kwargs):

        if not ax:
            fig, ax = plt.subplots(**kwargs)
        else:
            fig = plt.gcf()

        for b in self.bands:
            try:
                if not isinstance(lightcurve, type(None)):
                    ul_mask = np.array(lightcurve[f"{b}_{lum_key}{self.upper_limit_key}"]).astype(bool)
                    ax.errorbar(lightcurve.mean_mjd[~ul_mask], lightcurve[f"{b}{self.mean_key}_{lum_key}"][~ul_mask],
                                yerr=lightcurve[f"{b}_{lum_key}{self.rms_key}"][~ul_mask],
                                label=b, ls='', marker='s', c=self.band_plot_colors[b], markersize=4,
                                markeredgecolor='k', ecolor='k', capsize=2)
                    ax.scatter(lightcurve.mean_mjd[ul_mask], lightcurve[f"{b}{self.mean_key}_{lum_key}"][ul_mask],
                               marker='v', c=self.band_plot_colors[b], alpha=0.7, s=2)
                if not isinstance(unbinned_lc, type(None)):
                    ax.errorbar(unbinned_lc.mjd, unbinned_lc[f"{b}_{lum_key}"],
                                yerr=unbinned_lc[f"{b}_{lum_key}{self.error_key_ext}"],
                                label=f"{b} unbinned", ls='', marker='o', c=self.band_plot_colors[b], markersize=4,
                                alpha=0.3)
            except KeyError as e:
                raise KeyError(f"Could not find brightness key {e}!")

        if lum_key == 'mag':
            ylim = ax.get_ylim()
            ax.set_ylim([ylim[-1], ylim[0]])

        ax.set_xlabel('MJD')
        ax.set_ylabel(lum_key)
        ax.legend()

        if save:
            logger.debug(f"saving under {fn}")
            fig.savefig(fn)

        if interactive:
            return fig, ax
        else:
            plt.close()

    #####################################
    #  END MAKE PLOTTING FUNCTIONS      #
    ###########################################################################################################

    ###########################################################################################################
    #  START CALCULATE METADATA         #
    #####################################

    def _metadata_filename(self, service, chunk_number=None, jobID=None):

        warnings.warn("Separate metadata will be deprecated in timewise 0.3!", DeprecationWarning)

        if (chunk_number is None) and (jobID is None):
            return os.path.join(self.lightcurve_dir, f'metadata_{service}.json')
        elif (chunk_number is not None) and (jobID is None):
            return os.path.join(self.cache_dir, f'metadata_{service}{self._split_chunk_key}{chunk_number}.json')
        elif (chunk_number is not None) and (jobID is not None):
            return os.path.join(self.cache_dir, f'metadata_{service}{self._split_chunk_key}{chunk_number}_job{jobID}.json')
        else:
            raise NotImplementedError

    def _load_metadata(self, service, chunk_number=None, jobID=None, return_filename=False):
        fn = self._metadata_filename(service, chunk_number, jobID)
        try:
            logger.debug(f"loading {fn}")
            with open(fn, "r") as f:
                metadata = json.load(f)
            if return_filename:
                return metadata, fn
            return metadata
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_metadata(self, metadata, service, chunk_number=None, jobID=None, overwrite=False):
        fn = self._metadata_filename(service, chunk_number, jobID)

        if fn == self._metadata_filename(service):
            self._cached_final_products['metadata'][service] = metadata

        if not overwrite:
            try:
                old_metadata = self._load_metadata(service=service, chunk_number=chunk_number, jobID=jobID)
                logger.debug(f"Found {len(old_metadata)}. Combining")
                metadata = metadata.update(old_metadata)
            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new metadata.")

        logger.debug(f'saving under {fn}')
        with open(fn, "w") as f:
            json.dump(metadata, f)

    def load_metadata(self, service):
        """Load the metadata

        :param service: The service with which the lightcurves were downloaded
        :type service: str
        :return: the metadata
        :rtype: dict
        """
        if not service in self._cached_final_products['metadata']:
            self._cached_final_products['metadata'][service] = self._load_metadata(service)
        return self._cached_final_products['metadata'][service]

    def calculate_metadata(self, service, chunk_number=None, jobID=None, overwrite=True):
        """Calculates the metadata for all downloaded lightcurves.
         Results will be saved under

            </path/to/timewise/data/dir>/output/<base_name>/lightcurves/metadata_<service>.json

        :param service: the service with which the lightcurves were downloaded
        :type service: str
        :param chunk_number: the chunk number to use, default uses all chunks
        :type chunk_number: int
        :param jobID:  the job ID to use, default uses all lightcurves
        :type jobID: int
        :param overwrite: overwrite existing metadata file
        :type overwrite: bool
        """
        data_product = self._load_data_product(service, chunk_number, jobID)
        for ID, i_data_product in data_product.items():
            lc = pd.DataFrame.from_dict(i_data_product["timewise_lightcurve"])
            metadata = self.calculate_metadata_single(lc)
            data_product[ID]["timewise_metadata"] = metadata

        self._save_data_product(data_product, service, chunk_number, jobID, overwrite=overwrite)

    @abc.abstractmethod
    def calculate_metadata_single(self, lcs):
        """
        Calculates some properties of the lightcurves

        :param lcs: the lightcurve
        :type lcs: pandas.DataFrame
        """
        raise NotImplementedError

    #####################################
    #  END CALCULATE METADATA           #
    ###########################################################################################################
