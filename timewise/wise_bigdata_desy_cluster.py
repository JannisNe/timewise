import getpass
import glob
import os
import json
import subprocess
import math
import pickle
import queue
import threading
import argparse
import time
import warnings

import backoff
import shutil
import gc
import tqdm
import sys

from functools import cache
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo as vo
import traceback as tb
import gzip
import logging

from typing import List

from timewise.general import DATA_DIR_KEY, data_dir, bigdata_dir, backoff_hndlr
from timewise.wise_data_by_visit import WiseDataByVisit


logger = logging.getLogger(__name__)


class WISEDataDESYCluster(WiseDataByVisit):
    status_cmd = f'qstat -u {getpass.getuser()}'
    # finding the file that contains the setup function
    BASHFILE = os.getenv('TIMEWISE_DESY_CLUSTER_BASHFILE', os.path.expanduser('~/.bashrc'))

    def __init__(
            self,
            base_name,
            parent_sample_class,
            min_sep_arcsec,
            n_chunks,
            clean_outliers_when_binning=True,
            multiply_flux_error=True
    ):

        super().__init__(base_name=base_name, parent_sample_class=parent_sample_class, min_sep_arcsec=min_sep_arcsec,
                         n_chunks=n_chunks, clean_outliers_when_binning=clean_outliers_when_binning,
                         multiply_flux_error=multiply_flux_error)

        # set up cluster stuff
        self._status_output = None
        self.executable_filename = os.path.join(self.cluster_dir, "run_timewise.sh")
        self.submit_file_filename = os.path.join(self.cluster_dir, "submit_file.submit")
        self.job_id = None
        self._n_cluster_jobs_per_chunk = None
        self.cluster_jobID_map = None
        self.clusterJob_chunk_map = None
        self.cluster_info_file = os.path.join(self.cluster_dir, 'cluster_info.pkl')
        self._overwrite = True
        self._storage_dir = None

        # status attributes
        self.start_time = None
        self._total_tasks = None
        self._done_tasks = None

        self._tap_queue = queue.Queue()
        self._cluster_queue = queue.Queue()
        self._io_queue = queue.PriorityQueue()
        self._io_queue_done = queue.Queue()

    # ---------------------------------------------------------------------------------- #
    # START using gzip to compress the data when saving     #
    # ----------------------------------------------------- #

    def _data_product_filename(self, service, chunk_number=None, jobID=None, use_bigdata_dir=False):
        fn = super(WISEDataDESYCluster, self)._data_product_filename(service, chunk_number=chunk_number, jobID=jobID)

        if use_bigdata_dir:
            fn = fn.replace(data_dir, bigdata_dir)

        return fn + ".gz"

    def _load_data_product(
            self,
            service,
            chunk_number=None,
            jobID=None,
            return_filename=False,
            use_bigdata_dir=False
    ):
        fn = self._data_product_filename(
            service,
            chunk_number,
            jobID,
            use_bigdata_dir=use_bigdata_dir
        )

        logger.debug(f"loading {fn}")
        try:
            with gzip.open(fn, 'r') as fin:
                data_product = json.loads(fin.read().decode('utf-8'))
            if return_filename:
                return data_product, fn
            return data_product
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_data_product(
            self,
            data_product,
            service,
            chunk_number=None,
            jobID=None,
            overwrite=False,
            use_bigdata_dir=False
    ):
        fn = self._data_product_filename(
            service,
            chunk_number,
            jobID,
            use_bigdata_dir=use_bigdata_dir
        )
        logger.debug(f"saving {len(data_product)} new objects to {fn}")

        if fn == self._data_product_filename(service):
            self._cached_final_products['lightcurves'][service] = data_product

        if not overwrite:
            try:
                old_data_product = self._load_data_product(service=service, chunk_number=chunk_number, jobID=jobID,
                                                           use_bigdata_dir=use_bigdata_dir)

                if old_data_product is not None:
                    logger.debug(f"Found {len(old_data_product)}. Combining")
                    data_product = data_product.update(old_data_product)

            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new binned lightcurves.")

        with gzip.open(fn, 'w') as f:
            f.write(json.dumps(data_product).encode('utf-8'))

    def _load_lightcurves(
            self,
            service,
            chunk_number=None,
            jobID=None,
            return_filename=False,
            load_from_bigdata_dir=False
    ):
        fn = self._lightcurve_filename(service, chunk_number, jobID)

        if load_from_bigdata_dir:
            fn = fn.replace(data_dir, bigdata_dir)

        logger.debug(f"loading {fn}")
        try:
            with open(fn, "r") as f:
                lcs = json.load(f)
            if return_filename:
                return lcs, fn
            return lcs
        except FileNotFoundError:
            logger.warning(f"No file {fn}")


    def _load_metadata(
            self,
            service,
            chunk_number=None,
            jobID=None,
            return_filename=False,
            load_from_bigdata_dir=False
    ):
        fn = self._metadata_filename(service, chunk_number, jobID)

        if load_from_bigdata_dir:
            fn = fn.replace(data_dir, bigdata_dir)

        try:
            logger.debug(f"loading {fn}")
            with open(fn, "r") as f:
                metadata = json.load(f)
            if return_filename:
                return metadata, fn
            return metadata
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    # ----------------------------------------------------- #
    # END using gzip to compress the data when saving       #
    # ---------------------------------------------------------------------------------- #

    def get_sample_photometric_data(self, max_nTAPjobs=8, perc=1, tables=None, chunks=None,
                                    cluster_jobs_per_chunk=100, wait=5, remove_chunks=False,
                                    query_type='positional', overwrite=True,
                                    storage_directory=bigdata_dir,
                                    skip_download=False,
                                    skip_input=False):
        """
        An alternative to `get_photometric_data()` that uses the DESY cluster and is optimised for large datasets.

        :param max_nTAPjobs: The maximum number of TAP jobs active at the same time.
        :type max_nTAPjobs: int
        :param perc: The percentage of chunks to download
        :type perc: float
        :param tables: The tables to query
        :type tables: str or list-like
        :param chunks: chunks to download, default is all of the chunks
        :type chunks: list-like
        :param cluster_jobs_per_chunk: number of cluster jobs per chunk
        :type cluster_jobs_per_chunk: int
        :param wait: time in hours to wait after submitting TAP jobs
        :type wait: float
        :param remove_chunks: remove single chunk files after binning
        :type remove_chunks: bool
        :param query_type: 'positional': query photometry based on distance from object, 'by_allwise_id': select all photometry points within a radius of 50 arcsec with the corresponding AllWISE ID
        :type query_type: str
        :param overwrite: overwrite already existing lightcurves and metadata
        :type overwrite: bool
        :param storage_directory: move binned files and raw data here after work is done
        :type storage_directory: str
        :param skip_download: if True, assume data is already downloaded, only do binning in that case
        :type skip_download: bool
        :param skip_input: if True do not ask if data is correct before download
        :type skip_input: bool
        """

        # --------------------- set defaults --------------------------- #

        mag = True
        flux = True

        if tables is None:
            tables = [
                'AllWISE Multiepoch Photometry Table',
                'NEOWISE-R Single Exposure (L1b) Source Table'
            ]
        tables = np.atleast_1d(tables)

        if chunks is None:
            chunks = list(range(round(int(self.n_chunks * perc))))
        else:
            cm = [c not in self.chunk_map for c in chunks]
            if np.any(cm):
                raise ValueError(f"Chunks {np.array(chunks)[cm]} are not in chunk map. "
                                 f"Probably they are larger than the set chunk number of {self._n_chunks}")

        if remove_chunks:
            raise NotImplementedError("Removing chunks is not implemented yet!")

        if query_type not in self.query_types:
            raise ValueError(f"Unknown query type {query_type}! Choose one of {self.query_types}")

        service = 'tap'

        # set up queue
        self.queue = queue.Queue()

        # set up dictionary to store jobs in
        self.tap_jobs = {t: dict() for t in tables}

        logger.debug(f"Getting {perc * 100:.2f}% of lightcurve chunks ({len(chunks)}) via {service} "
                     f"in {'magnitude' if mag else ''} {'flux' if flux else ''} "
                     f"from {tables}\nskipping download: {skip_download}")

        if not skip_input:
            input('Correct? [hit enter] ')

        # --------------------------- set up cluster info --------------------------- #

        self.n_cluster_jobs_per_chunk = cluster_jobs_per_chunk
        cluster_time_s = max(len(self.parent_sample.df) / self._n_chunks / self.n_cluster_jobs_per_chunk, 59 * 60)
        if cluster_time_s > 24 * 3600:
            raise ValueError(f"cluster time per job would be longer than 24h! "
                             f"Choose more than {self.n_cluster_jobs_per_chunk} jobs per chunk!")

        cluster_time = time.strftime('%H:%M:%S', time.gmtime(cluster_time_s))
        self.clear_cluster_log_dir()
        self._save_cluster_info()
        self._overwrite = overwrite
        self._storage_dir = storage_directory

        # --------------------------- starting threads --------------------------- #

        tap_threads = [threading.Thread(target=self._tap_thread, daemon=True, name=f"TAPThread{_}")
                       for _ in range(max_nTAPjobs)]
        cluster_threads = [threading.Thread(target=self._cluster_thread, daemon=True, name=f"ClusterThread{_}")
                           for _ in range(max_nTAPjobs)]
        io_thread = threading.Thread(target=self._io_thread, daemon=True, name="IOThread")
        status_thread = threading.Thread(target=self._status_thread, daemon=True, name='StatusThread')

        for t in tap_threads + cluster_threads + [io_thread]:
            logger.debug('starting thread')
            t.start()

        logger.debug(f'started {len(tap_threads)} TAP threads and {len(cluster_threads)} cluster threads.')

        # --------------------------- filling queue with tasks --------------------------- #

        self.start_time = time.time()
        self._total_tasks = len(chunks)
        self._done_tasks = 0

        for c in chunks:
            if not skip_download:
                self._tap_queue.put((tables, c, wait, mag, flux, cluster_time, query_type))
            else:
                self._cluster_queue.put((cluster_time, c))

        status_thread.start()

        # --------------------------- wait for completion --------------------------- #

        logger.debug(f'added {self._tap_queue.qsize()} tasks to tap queue')
        self._tap_queue.join()
        logger.debug('TAP done')
        self._cluster_queue.join()
        logger.debug('cluster done')

    @backoff.on_exception(
        backoff.expo,
        vo.dal.exceptions.DALServiceError,
        giveup=WiseDataByVisit._give_up_tap,
        max_tries=50,
        on_backoff=backoff_hndlr
    )
    def _wait_for_job(self, t, i):
        logger.info(f"Waiting on {i}th query of {t} ........")
        _job = self.tap_jobs[t][i]
        _job.wait()
        logger.info(f'{i}th query of {t}: Done!')

    def _get_results_from_job(self, t, i):
        logger.debug(f"getting results for {i}th query of {t} .........")
        _job = self.tap_jobs[t][i]
        lightcurve = _job.fetch_result().to_table().to_pandas()
        fn = self._chunk_photometry_cache_filename(t, i)
        table_nice_name = self.get_db_name(t, nice=True)
        logger.debug(f"{i}th query of {table_nice_name}: saving under {fn}")
        cols = dict(self.photometry_table_keymap[table_nice_name]['mag'])
        cols.update(self.photometry_table_keymap[table_nice_name]['flux'])

        if 'allwise' in t:
            cols['cntr_mf'] = 'allwise_cntr'

        lightcurve.rename(columns=cols).to_csv(fn)
        return

    def _io_queue_hash(self, method_name, args):
        return f"{method_name}_{args}"

    def _wait_for_io_task(self, method_name, args):
        h = self._io_queue_hash(method_name, args)
        logger.debug(f"waiting on io-task {h}")

        while True:
            _io_queue_done = list(self._io_queue_done.queue)
            if h in _io_queue_done:
                break

            time.sleep(30)

        logger.debug(f"{h} done!")

    def _io_thread(self):
        logger.debug("started in-out thread")
        while True:
            priority, method_name, args = self._io_queue.get(block=True)
            logger.debug(f"executing {method_name} with arguments {args} (priority {priority})")

            try:
                self.__getattribute__(method_name)(*args)
                self._io_queue_done.put(self._io_queue_hash(method_name, args))
            except Exception as e:
                msg = (
                    f"#################################################################\n"
                    f"                !!!     ATTENTION     !!!                 \n"
                    f" ----------------- {method_name}({args}) ---------------- \n"
                    f"                      AN ERROR OCCURED                    \n"
                    f"\n{''.join(tb.format_exception(None, e, e.__traceback__))}\n\n"
                    f"putting {method_name}({args}) back into IO-queue\n"
                    f"#################################################################\n"
                )
                logger.error(msg)
                self._io_queue.put((priority, method_name, args))
            finally:
                self._io_queue.task_done()
                gc.collect()

    def _tap_thread(self):
        logger.debug(f'started tap thread')
        while True:
            tables, chunk, wait, mag, flux, cluster_time, query_type = self._tap_queue.get(block=True)
            logger.debug(f'querying IRSA for chunk {chunk}')

            submit_to_cluster = True

            for i in range(len(tables) + 1):

                # -----------   submit jobs for chunk i via the IRSA TAP  ---------- #
                if i < len(tables):
                    t = tables[i]
                    submit_method = "_submit_job_to_TAP"
                    submit_args = [chunk, t, mag, flux, query_type]
                    self._io_queue.put((1, submit_method, submit_args))
                    self._wait_for_io_task(submit_method, submit_args)

                # --------------  get results of TAP job for chunk i-1 ------------- #
                if i > 0:
                    t_before = tables[i - 1]

                    if self.tap_jobs[t_before][chunk].phase == "COMPLETED":
                        result_method = "_get_results_from_job"
                        result_args = [t_before, chunk]
                        self._io_queue.put((2, result_method, result_args))
                        self._wait_for_io_task(result_method, result_args)

                    else:
                        logger.warning(
                            f"No completion for {chunk}th query of {t_before}! "
                            f"Phase is {self.tap_jobs[t_before][chunk].phase}!"
                        )
                        submit_to_cluster = False

                # ---------------   wait for the TAP job of chunk i  -------------- #
                if i < len(tables):
                    t = tables[i]
                    logger.info(f'waiting for {wait} hours')
                    time.sleep(wait * 3600)

                    try:
                        self._wait_for_job(t, chunk)
                    except vo.dal.exceptions.DALServiceError:
                        logger.warning(f"could not wait for {chunk}th query of {t}! Not submitting to cluster.")
                        # mark task as done and move on without submission to cluster
                        submit_to_cluster = False
                        continue

            self._tap_queue.task_done()
            if submit_to_cluster:
                self._cluster_queue.put((cluster_time, chunk))

            gc.collect()

    def _move_file_to_storage(self, filename):
        dst_fn = filename.replace(data_dir, self._storage_dir)

        dst_dir = os.path.dirname(dst_fn)
        if not os.path.isdir(dst_dir):
            logger.debug(f"making directory {dst_dir}")
            os.makedirs(dst_dir)

        logger.debug(f"copy {filename} to {dst_fn}")

        try:
            shutil.copy2(filename, dst_fn)

            if os.path.getsize(filename) == os.path.getsize(dst_fn):
                logger.debug(f"copy successful, removing {filename}")
                os.remove(filename)
            else:
                logger.warning(f"copy from {filename} to {dst_fn} gone wrong! Not removing source.")

        except FileNotFoundError as e:
            logger.warning(f"FileNotFoundError: {e}!")

    def _cluster_thread(self):
        logger.debug(f'started cluster thread')
        while True:
            cluster_time, chunk = self._cluster_queue.get(block=True)

            logger.info(f'got all TAP results for chunk {chunk}. submitting to cluster')
            job_id = self.submit_to_cluster(node_memory='40G',
                                            single_chunk=chunk)

            if not job_id:
                logger.warning(f"could not submit {chunk} to cluster! Try later")
                self._cluster_queue.put((cluster_time, chunk))
                self._cluster_queue.task_done()

            else:
                logger.debug(f'waiting for chunk {chunk} (Cluster job {job_id})')
                self.wait_for_job()
                logger.debug(f'cluster done for chunk {chunk} (Cluster job {job_id}).')

                log_files = glob.glob(f"./{self.job_id}_*")
                log_files_abs = [os.path.abspath(p) for p in log_files]
                logger.debug(f"moving {len(log_files_abs)} log files to {self.cluster_log_dir}")
                for f in log_files_abs:
                    shutil.move(f, self.cluster_log_dir)

                logger.debug(f'Start combining')

                try:
                    self._combine_data_products('tap', chunk_number=chunk, remove=True, overwrite=self._overwrite)

                    if self._storage_dir:
                        filenames_to_move = [
                            self._data_product_filename(service='tap', chunk_number=chunk),
                        ]

                        for t in self.photometry_table_keymap.keys():
                            filenames_to_move.append(self._chunk_photometry_cache_filename(t, chunk))

                        for fn in filenames_to_move:
                            self._move_file_to_storage(fn)

                finally:
                    self._cluster_queue.task_done()
                    self._done_tasks += 1
                    gc.collect()

    def _status_thread(self):
        logger.debug('started status thread')
        while True:
            n_tap_tasks_queued = self._tap_queue.qsize()
            n_cluster_tasks_queued = self._cluster_queue.qsize()
            n_remaining = self._total_tasks - self._done_tasks
            elapsed_time = time.time() - self.start_time
            time_per_task = elapsed_time / self._done_tasks if self._done_tasks > 0 else np.nan
            remaining_time = n_remaining * time_per_task

            msg = f"\n-----------------     STATUS     -----------------\n" \
                  f"\ttasks in TAP queue:_______{n_tap_tasks_queued}\n" \
                  f"\ttasks in cluster queue:___{n_cluster_tasks_queued}\n" \
                  f"\tperformed io tasks:_______{len(list(self._io_queue_done.queue))}\n" \
                  f"\tdone total:_______________{self._done_tasks}/{self._total_tasks}\n" \
                  f"\truntime:__________________{elapsed_time/3600:.2f} hours\n" \
                  f"\tremaining:________________{remaining_time/3600:.2f} hours"

            logger.info(msg)
            time.sleep(5*3600)

    # ----------------------------------------------------------------------------------- #
    # START using cluster for downloading and binning      #
    # ---------------------------------------------------- #

    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        OSError,
        max_time=2*3600,
        on_backoff=backoff_hndlr,
        jitter=backoff.full_jitter,
    )
    def _execute_bash_command(cmd):
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as process:
            msg = process.stdout.read().decode()
            process.terminate()
        return msg

    @staticmethod
    def get_condor_status():
        """
        Queries condor to get cluster status.
        :return: str, output of query command
        """
        cmd = "condor_q"
        return WISEDataDESYCluster._execute_bash_command(cmd)

    def collect_condor_status(self):
        """Gets the condor status and saves it to private attribute"""
        self._status_output = self.get_condor_status()

    @property
    def condor_status(self):
        """
        Get the status of jobs running on condor.
        :return: number of jobs that are done, running, waiting, total, held
        """
        status_list = [
            [y for y in ii.split(" ") if y]
            for ii in self._status_output.split("\n")[4:-6]
        ]
        done = running = waiting = total = held = None

        for li in status_list:
            if li[2] == self.job_id:
                done, running, waiting = li[5:8]
                held = 0 if len(li) == 10 else li[8]
                total = li[-2]

        return done, running, waiting, total, held

    def wait_for_job(self):
        """
        Wait until the cluster job is done
        """

        if self.job_id:
            logger.info("waiting for job with ID " + str(self.job_id))
            time.sleep(5)

            self.collect_condor_status()
            j = 0
            while not np.all(np.array(self.condor_status) == None):
                d, r, w, t, h = self.condor_status
                logger.info(
                    f"{time.asctime(time.localtime())} - Job{self.job_id}: "
                    f"{d} done, {r} running, {w} waiting, {h} held of total {t}"
                )
                j += 1
                if j > 7:
                    logger.info(self._status_output)
                    j = 0
                time.sleep(90)
                self.collect_condor_status()

            logger.info("Done waiting for job with ID " + str(self.job_id))

        else:
            logger.info(f"No Job ID!")

    @property
    def n_cluster_jobs_per_chunk(self):
        return self._n_cluster_jobs_per_chunk

    @n_cluster_jobs_per_chunk.setter
    def n_cluster_jobs_per_chunk(self, value):
        self._n_cluster_jobs_per_chunk = value

        if value:
            n_jobs = self.n_chunks * int(value)
            logger.debug(f'setting {n_jobs} jobs.')
            self.cluster_jobID_map = np.zeros(len(self.parent_sample.df), dtype=int)
            self.clusterJob_chunk_map = pd.DataFrame(columns=['chunk_number'])

            for chunk_number in range(self.n_chunks):
                indices = np.where(self.chunk_map == chunk_number)[0]
                N_inds_per_job = int(math.ceil(len(indices) / self._n_cluster_jobs_per_chunk))
                for j in range(self._n_cluster_jobs_per_chunk):
                    job_nr = chunk_number*self._n_cluster_jobs_per_chunk + j + 1
                    self.clusterJob_chunk_map.loc[job_nr] = [chunk_number]
                    start_ind = j * N_inds_per_job
                    end_ind = start_ind + N_inds_per_job
                    self.cluster_jobID_map[indices[start_ind:end_ind]] = job_nr

        else:
            logger.warning(f'Invalid value for n_cluster_jobs_per_chunk: {value}')

    def _get_chunk_number_for_job(self, jobID):
        chunk_number = self.clusterJob_chunk_map.loc[jobID, 'chunk_number']
        return chunk_number

    def _save_cluster_info(self):
        logger.debug(f"writing cluster info to {self.cluster_info_file}")
        with open(self.cluster_info_file, "wb") as f:
            pickle.dump((self.cluster_jobID_map, self.clusterJob_chunk_map, self.clean_outliers_when_binning), f)

    def _load_cluster_info(self):
        logger.debug(f"loading cluster info from {self.cluster_info_file}")
        with open(self.cluster_info_file, "rb") as f:
            self.cluster_jobID_map, self.clusterJob_chunk_map, self.clean_outliers_when_binning = pickle.load(f)

    def clear_cluster_log_dir(self):
        """
        Clears the directory where cluster logs are stored
        """
        fns = os.listdir(self.cluster_log_dir)
        for fn in fns:
            os.remove(os.path.join(self.cluster_log_dir, fn))

    def make_executable_file(self):
        """
        Produces the executable that will be submitted to the NPX cluster.
        """
        logging_level = logger.getEffectiveLevel()
        script_fn = os.path.realpath(__file__)

        txt = (
            f'{sys.executable} {script_fn} '
            f'--logging_level {logging_level} '
            f'--base_name {self.base_name} '
            f'--min_sep_arcsec {self.min_sep.to("arcsec").value} '
            f'--n_chunks {self._n_chunks} '
            f'--job_id $1 '
        )

        logger.debug("writing executable to " + self.executable_filename)
        with open(self.executable_filename, "w") as f:
            f.write(txt)

    def get_submit_file_filename(self, ids):
        """
        Get the filename of the submit file for given job ids

        :param ids: list of job ids
        :type ids: list
        :return: filename
        :rtype: str
        """
        ids = np.atleast_1d(ids)
        ids_string = f"{min(ids)}-{max(ids)}"
        return os.path.join(self.cluster_dir, f"ids{ids_string}.submit")

    def make_submit_file(
            self,
            job_ids: (int, List[int]),
            node_memory: str = '8G',
    ):
        """
        Produces the submit file that will be submitted to the NPX cluster.

        :param job_ids: The job ID or list of job IDs to submit
        :type job_ids: int or list of ints
        :param node_memory: The amount of memory to request for each node
        :type node_memory: str
        """

        q = "1 job_id in " + ", ".join(np.atleast_1d(job_ids).astype(str))

        text = (
            f"executable = {self.executable_filename} \n"
            f"environment = \"TIMEWISE_DATA={data_dir} TIMEWISE_BIGDATA={bigdata_dir}\" \n"
            f"log = $(cluster)_$(process)job.log \n"
            f"output = $(cluster)_$(process)job.out \n"
            f"error = $(cluster)_$(process)job.err \n"
            f"should_transfer_files   = YES \n"
            f"when_to_transfer_output = ON_EXIT \n"
            f"arguments = $(job_id) \n"
            f"RequestMemory = {node_memory} \n"
            f"\n"
            f"queue {q}"
        )

        fn = self.get_submit_file_filename(job_ids)
        logger.debug("writing submitfile at " + fn)
        with open(fn, "w") as f:
            f.write(text)

    def submit_to_cluster(self, node_memory, single_chunk=None):
        """
        Submit jobs to cluster

        :param node_memory: memory per node
        :type node_memory: str
        :param single_chunk: number of single chunk to run on the cluster
        :type single_chunk: int
        :return: ID of the cluster job
        :rtype: int
        """

        if isinstance(single_chunk, type(None)):
            _start_id = 1
            _end_id = int(self.n_chunks*self.n_cluster_jobs_per_chunk)
        else:
            _start_id = int(single_chunk*self.n_cluster_jobs_per_chunk) + 1
            _end_id = int(_start_id + self.n_cluster_jobs_per_chunk) - 1

        ids = list(range(_start_id, _end_id + 1))

        # make data_product files, storing essential info from parent_sample
        for jobID in ids:
            indices = self.parent_sample.df.index[self.cluster_jobID_map == jobID]
            logger.debug(f"starting data_product for {len(indices)} objects.")
            data_product = self._start_data_product(parent_sample_indices=indices)
            chunk_number = self._get_chunk_number_for_job(jobID)
            self._save_data_product(data_product, service="tap", chunk_number=chunk_number, jobID=jobID)

        parentsample_class_pickle = os.path.join(self.cluster_dir, 'parentsample_class.pkl')
        logger.debug(f"pickling parent sample class to {parentsample_class_pickle}")
        with open(parentsample_class_pickle, "wb") as f:
            pickle.dump(self.parent_sample_class, f)

        self.make_executable_file()
        self.make_submit_file(job_ids=ids, node_memory=node_memory)

        submit_cmd = 'condor_submit ' + self.get_submit_file_filename(ids)
        logger.info(f"{time.asctime(time.localtime())}: {submit_cmd}")

        try:
            msg = self._execute_bash_command(submit_cmd)
            logger.info(str(msg))
            job_id = str(msg).split("cluster ")[-1].split(".")[0]
            logger.info(f"Running on cluster with ID {job_id}")
            self.job_id = job_id
            return job_id

        except OSError:
            return

    def run_cluster(self, node_memory, service):
        """
        Run the DESY cluster

        :param node_memory: memory per node
        :type node_memory: str
        :param service: service to use for querying the data
        :type service: str
        """

        self.clear_cluster_log_dir()
        self._save_cluster_info()
        self.submit_to_cluster(node_memory)
        self.wait_for_job()
        for c in range(self.n_chunks):
            self._combine_data_products(service, chunk_number=c, remove=True, overwrite=True)

    # ---------------------------------------------------- #
    # END using cluster for downloading and binning        #
    # ----------------------------------------------------------------------------------- #

    ###########################################################################################################
    # START MAKE PLOTTING FUNCTIONS     #
    #####################################

    def plot_lc(
            self,
            parent_sample_idx,
            service='tap',
            plot_unbinned=False,
            plot_binned=True,
            interactive=False,
            fn=None,
            ax=None,
            save=True,
            lum_key='flux_density',
            load_from_bigdata_dir=False,
            **kwargs
    ):
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
        :type lum_key: str
        :param load_from_bigdata_dir: load from the the big data storage directory
        :type load_from_bigdata_dir: bool
        :param kwargs: any additional kwargs will be passed on to `matplotlib.pyplot.subplots()`
        :return: the matplotlib.Figure and matplotlib.Axes if `interactive=True`
        """

        logger.debug(f"loading binned lightcurves")

        _get_unbinned_lcs_fct = self._get_unbinned_lightcurves \
            if service == 'tap' else self._get_unbinned_lightcurves_gator

        wise_id = self.parent_sample.df.loc[int(parent_sample_idx), self.parent_wise_source_id_key]
        if isinstance(wise_id, float) and not np.isnan(wise_id):
            wise_id = int(wise_id)
        logger.debug(f"{wise_id} for {parent_sample_idx}")

        _chunk_number = self._get_chunk_number(parent_sample_index=parent_sample_idx)
        data_product = self._load_data_product(
            service,
            chunk_number=_chunk_number,
            use_bigdata_dir=load_from_bigdata_dir
        )
        lc = pd.DataFrame.from_dict(data_product[parent_sample_idx]["timewise_lightcurve"])

        if plot_unbinned:

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

    # --------------------------------------------------------------------------------------
    #             START Chi2 plots
    # -------------------------------------------

    @cache
    def get_red_chi2(self, chunk, lum_key, use_bigdata_dir):
        # TODO: add doc

        logger.info(f"extracting info for chunk {chunk}")
        data_product = self._load_data_product(
            service="tap",
            chunk_number=chunk,
            use_bigdata_dir=use_bigdata_dir
        )

        chi2_val = {b: dict() for b in self.bands}

        for b in self.bands:
            key1 = f"{b}_chi2_to_med{lum_key}"
            key2 = f"{b}_N_datapoints{lum_key}"
            key3 = f"{b}_median{lum_key}"
            logger.debug(f"{key1}, {key2}")

            for i, idata_product in tqdm.tqdm(
                    data_product.items(),
                    total=len(data_product),
                    desc="collecting chi2 values"
            ):
                if "timewise_metadata" in idata_product:
                    imetadata = idata_product["timewise_metadata"]

                    if (key1 in imetadata) and (key2 in imetadata):
                        ndof = (imetadata[key2] - 1)
                        v = {
                            "chi2": imetadata[key1] / ndof if ndof > 0 else np.nan,
                            "med_lum": imetadata[key3],
                            "N_datapoints": imetadata[key2]
                        }
                        chi2_val[b][i] = v

        return {b: pd.DataFrame.from_dict(chi2_val[b], orient='index') for b in self.bands}

    def make_chi2_plot(
            self,
            index_mask=None,
            chunks=None,
            load_from_bigdata_dir=False,
            lum_key="_flux_density",
            interactive=False,
            save=False,
            nbins=100,
            cumulative=True,
            upper_bound=4
    ):

        if chunks is None:
            chunks = list(range(self.n_chunks))

        chi2_data_list = [self.get_red_chi2(chunk, lum_key, load_from_bigdata_dir) for chunk in chunks]
        chi2_data = {b: pd.concat([d[b] for d in chi2_data_list]) for b in self.bands}

        N_datapoints = set.intersection(*[set(df["N_datapoints"].unique()) for b, df in chi2_data.items()])

        res = list()

        for n in N_datapoints:

            if n == 1:
                continue

            chi2_df_sel = {b: df[df["N_datapoints"] == n]["chi2"] for b, df in chi2_data.items()}

            logger.info(f"making chi2 histogram for lightcurves with {n} datapoints")

            fig, axs = plt.subplots(
                ncols=len(self.bands),
                figsize=(10, 5),
                sharey="all",
                sharex="all"
            )

            index_colors = {k: f"C{i}" for i, k in enumerate(index_mask.keys())} if index_mask is not None else None

            x = np.linspace(0, upper_bound, nbins)
            x = np.concatenate([x, [1e6]])

            for ax, band in zip(axs, self.bands):
                h, b, _ = ax.hist(
                    chi2_df_sel[band].values.flatten(),
                    label="all",
                    density=True,
                    cumulative=cumulative,
                    color="k",
                    bins=x,
                    alpha=0.4
                )
                bmids = (b[1:] + b[:-1]) / 2

                # if cumulative then also calculate the histogram of the PDF
                # this will be used later to calculate the goodness of fit to the F- and Chi2-distribution
                hpdf = h if not cumulative else \
                    np.histogram(chi2_df_sel[band].values.flatten(), bins=x, density=True)[0]
                nonzero_m = hpdf > 0

                # we need the absolute histogram numbers to calculate the uncsertainties of the density bins
                # The uncertainty of the density bin d_i is
                #
                #       u_di = u_ci * sum_{j not i}(c_j) / [(sum_{j}c_j)^2 * (b_{i+1} - b_{i})]
                #
                # where u_ci = sqrt(c_i) is the uncertainty of the counts bin c_i
                #
                h_abs = np.histogram(chi2_df_sel[band].values.flatten(), bins=x, density=False)[0][nonzero_m]
                h_abs_sum = np.sum(h_abs)
                h_sum_not_i = h_abs_sum - h_abs
                u_density = np.sqrt(h_abs) * h_sum_not_i / (h_abs_sum ** 2 * (np.diff(x))[nonzero_m])

                if index_mask is not None:
                    for label, indices in index_mask.items():
                        _indices = chi2_df_sel[band].index.intersection(indices)
                        ax.hist(
                            chi2_df_sel[band].loc[_indices].values.flatten(),
                            label=label,
                            histtype="step",
                            bins=b,
                            density=True,
                            cumulative=cumulative,
                            color=index_colors[label]
                        )

                # select non-NaN's and values below `upper_bound`
                sel = chi2_df_sel[band][(~chi2_df_sel[band].isna()) & (chi2_df_sel[band] < upper_bound)]
                if len(sel) > 0:

                    # fit an F-distribution
                    fpars = f.fit(sel, n-1, 1e5, f0=1, floc=0)
                    frozenf = f(*fpars)
                    fpdf = frozenf.pdf

                    # if cumulative then draw the CDF instead of the PDF
                    ffunc = frozenf.cdf if cumulative else fpdf

                    # To see how well the distribution fits the data we'll caluclate the chi2
                    # to the PDF (not to the CDF because the bins in CDF are correlated)
                    ndof_fit = len(bmids[nonzero_m]) - 2
                    F_chi2fit = sum((hpdf[nonzero_m] - fpdf(bmids[nonzero_m])) ** 2 / u_density**2) / ndof_fit

                    # plot the fitted distribution
                    ax.plot(x, ffunc(x), color='k', ls="-.",
                            label=(
                                f"fitted F-distribution ($\chi ^2$={F_chi2fit:.2f})\n "
                                f"ndof1={fpars[0]:.2f}, ndof2={fpars[1]:.2f}, scale={fpars[-1]:.2f}"
                            )
                            )

                    # fit an chi2 distribution
                    chi2fit_pars = chi2.fit(sel, n - 1, f0=n - 1, floc=0)
                    frozen_chi2fit = chi2(*chi2fit_pars)
                    frozen_chi2fit_pdf = frozen_chi2fit.pdf

                    # if cumulative then draw the CDF instead of the PDF
                    chi2fit_func = frozen_chi2fit.cdf if cumulative else frozen_chi2fit_pdf

                    # To see how well the distribution fits the data we'll caluclate the chi2
                    # to the PDF (not to the CDF because the bins in CDF are correlated)
                    ndof_fit = len(bmids[nonzero_m]) - 1
                    chi2fit_chi2fit = sum((hpdf[nonzero_m] - frozen_chi2fit_pdf(bmids[nonzero_m])) ** 2 /
                                          u_density**2) / ndof_fit

                    # plot the fitted distribution
                    ax.plot(x, chi2fit_func(x), color='k', ls=":",
                            label=(
                                f"fitted $\chi^2$-distribution ($\chi ^2$={chi2fit_chi2fit:.2f})\n "
                                f"fixed ndof={chi2fit_pars[0]:.2f}, scale={chi2fit_pars[-1]:.2f}"
                            )
                            )

                # we will also show the expected chi2 distribution
                pars_expected = (n - 1, 0, 1 / (n - 1))
                chi2_expected = chi2(*pars_expected)
                r = chi2_expected.cdf(x) if cumulative else chi2_expected.pdf(x)
                ax.plot(x, r, color="k", ls="--",
                        label=f"expected $\chi ^2$\n ndof: {n - 1:.1f} scale: {1/(n - 1):.2f}")

                ax.legend()
                ax.set_xlabel("$\chi^2$ " + band)
                ax.set_xlim(0, upper_bound)
                fig.suptitle(f"{n} datapoints")
                fig.tight_layout()

            if save:
                kind = "cdf" if cumulative else "pdf"
                fn = os.path.join(self.plots_dir, f"chi2_plots", lum_key, f"{n}_datapoints_{kind}.pdf")
                d = os.path.dirname(fn)
                if not os.path.isdir(d):
                    os.makedirs(d)
                logger.debug(f"saving under {fn}")
                fig.savefig(fn)

            if interactive:
                res.append((fig, axs))
            else:
                plt.close()

        if interactive:
            return res

    # -------------------------------------------
    #             END Chi2 plots
    # --------------------------------------------------------------------------------------

    #####################################
    # END MAKE PLOTTING FUNCTIONS       #
    ###########################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--base_name', type=str)
    parser.add_argument('--min_sep_arcsec', type=float)
    parser.add_argument('--n_chunks', type=int)
    parser.add_argument('--clear_unbinned', type=bool, default=False)
    parser.add_argument('--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    try:
        logging_level = int(cfg.logging_level)
    except ValueError:
        logging_level = cfg.logging_level.upper()

    logging.getLogger("timewise").setLevel(logging_level)
    logger = logging.getLogger("timewise.main")
    logger.info(json.dumps(vars(cfg), indent=4))

    wd = WISEDataDESYCluster(base_name=cfg.base_name,
                             min_sep_arcsec=cfg.min_sep_arcsec,
                             n_chunks=cfg.n_chunks,
                             parent_sample_class=None)
    wd._load_cluster_info()
    wd.clear_unbinned_photometry_when_binning = cfg.clear_unbinned
    chunk_number = wd._get_chunk_number_for_job(cfg.job_id)

    wd._subprocess_select_and_bin(service='tap', chunk_number=chunk_number, jobID=cfg.job_id)
    wd.calculate_metadata(service='tap', chunk_number=chunk_number, jobID=cfg.job_id)
