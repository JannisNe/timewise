import getpass
import os
import json
import subprocess
import math
import pickle
import queue
import threading
import argparse
import time
import backoff
import shutil
import gc
import numpy as np
import pandas as pd
import pyvo as vo
import traceback as tb
import gzip
import logging

from timewise.general import DATA_DIR_KEY, data_dir, bigdata_dir, backoff_hndlr
from timewise.wise_data_by_visit import WiseDataByVisit


logger = logging.getLogger(__name__)


class WISEDataDESYCluster(WiseDataByVisit):
    status_cmd = f'qstat -u {getpass.getuser()}'
    # finding the file that contains the setup function tde_catalogue
    BASHFILE = os.getenv('TIMEWISE_DESY_CLUSTER_BASHFILE', os.path.expanduser('~/.bashrc'))

    def __init__(self, base_name, parent_sample_class, min_sep_arcsec, n_chunks):

        super().__init__(base_name=base_name,
                         parent_sample_class=parent_sample_class,
                         min_sep_arcsec=min_sep_arcsec,
                         n_chunks=n_chunks)

        # set up cluster stuff
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
            job_id = self.submit_to_cluster(cluster_cpu=1,
                                            cluster_h=cluster_time,
                                            cluster_ram='40G',
                                            tables=None,
                                            service='tap',
                                            single_chunk=chunk)

            if not job_id:
                logger.warning(f"could not submit {chunk} to cluster! Try later")
                self._cluster_queue.put((cluster_time, chunk))
                self._cluster_queue.task_done()

            else:
                logger.debug(f'waiting for chunk {chunk} (Cluster job {job_id})')
                self.wait_for_job(job_id)
                logger.debug(f'cluster done for chunk {chunk} (Cluster job {job_id}). Start combining')

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
    def _qstat_output(qstat_command):
        """return the output of the qstat_command"""
        # start a subprocess to query the cluster
        return str(WISEDataDESYCluster._execute_bash_command(qstat_command))

    @staticmethod
    def _get_ids(qstat_command):
        """Takes a command that queries the DESY cluster and returns a list of job IDs"""
        st = WISEDataDESYCluster._qstat_output(qstat_command)
        # If the output is an empty string there are no tasks left
        if st == '':
            ids = list()
        else:
            # Extract the list of job IDs
            ids = np.array([int(s.split(' ')[1]) for s in st.split('\n')[2:-1]])
        return ids

    def _ntasks_from_qstat_command(self, qstat_command, job_id):
        """Returns the number of tasks from the output of qstat_command"""
        # get the output of qstat_command
        ids = self._get_ids(qstat_command)
        ntasks = 0 if len(ids) == 0 else len(ids[ids == job_id])
        return ntasks

    def _ntasks_total(self, job_id):
        """Returns the total number of tasks"""
        return self._ntasks_from_qstat_command(self.status_cmd, job_id)

    def _ntasks_running(self, job_id):
        """Returns the number of running tasks"""
        return self._ntasks_from_qstat_command(self.status_cmd + " -s r", job_id)

    def wait_for_job(self, job_id=None):
        """
        Wait until the cluster job is done

        :param job_id: the ID of the cluster job, if `None` use `self.job_ID`
        :type job_id: int
        """
        _job_id = job_id if job_id else self.job_id

        if _job_id:
            logger.info(f'waiting on job {_job_id}')
            time.sleep(10)
            i = 31
            j = 6
            while self._ntasks_total(_job_id) != 0:
                if i > 30:
                    logger.info(f'{time.asctime(time.localtime())} - Job{_job_id}:'
                                f' {self._ntasks_total(_job_id)} entries in queue. '
                                f'Of these, {self._ntasks_running(_job_id)} are running tasks, and '
                                f'{self._ntasks_total(_job_id) - self._ntasks_running(_job_id)} '
                                f'are tasks still waiting to be executed.')
                    i = 0
                    j += 1

                if j > 7:
                    logger.info(self._qstat_output(self.status_cmd))
                    j = 0

                time.sleep(30)
                i += 1

            logger.info('cluster is done')

        else:
            logger.info(f'No Job ID!')

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
            pickle.dump((self.cluster_jobID_map, self.clusterJob_chunk_map), f)

    def _load_cluster_info(self):
        logger.debug(f"loading cluster info from {self.cluster_info_file}")
        with open(self.cluster_info_file, "rb") as f:
            self.cluster_jobID_map, self.clusterJob_chunk_map = pickle.load(f)

    def clear_cluster_log_dir(self):
        """
        Clears the directory where cluster logs are stored
        """
        fns = os.listdir(self.cluster_log_dir)
        for fn in fns:
            os.remove(os.path.join(self.cluster_log_dir, fn))

    def _make_cluster_script(self, cluster_h, cluster_ram, tables, service):
        script_fn = os.path.realpath(__file__)

        if tables:
            tables = np.atleast_1d(tables)
            tables = [self.get_db_name(t, nice=False) for t in tables]
            tables_str = f"--tables {' '.join(tables)} \n"
        else:
            tables_str = '\n'

        text = "#!/bin/zsh \n" \
               "## \n" \
               "##(otherwise the default shell would be used) \n" \
               "#$ -S /bin/zsh \n" \
               "## \n" \
               "##(the running time for this job) \n" \
              f"#$ -l h_cpu={cluster_h} \n" \
               "#$ -l h_rss=" + str(cluster_ram) + "\n" \
               "## \n" \
               "## \n" \
               "##(send mail on job's abort) \n" \
               "#$ -m a \n" \
               "## \n" \
               "##(stderr and stdout are merged together to stdout) \n" \
               "#$ -j y \n" \
               "## \n" \
               "## name of the job \n" \
               "## -N TDE Catalogue download \n" \
               "## \n" \
               "##(redirect output to:) \n" \
               f"#$ -o /dev/null \n" \
               "## \n" \
               "sleep $(( ( RANDOM % 60 )  + 1 )) \n" \
               'exec > "$TMPDIR"/${JOB_ID}_${SGE_TASK_ID}_stdout.txt ' \
               '2>"$TMPDIR"/${JOB_ID}_${SGE_TASK_ID}_stderr.txt \n' \
              f'source {WISEDataDESYCluster.BASHFILE} \n' \
              f'export {DATA_DIR_KEY}={data_dir} \n' \
               'export O=1 \n' \
              f'python {script_fn} ' \
               f'--logging_level DEBUG ' \
               f'--base_name {self.base_name} ' \
               f'--min_sep_arcsec {self.min_sep.to("arcsec").value} ' \
               f'--n_chunks {self._n_chunks} ' \
               f'--job_id $SGE_TASK_ID ' \
               f'{tables_str}' \
               'cp $TMPDIR/${JOB_ID}_${SGE_TASK_ID}_stdout.txt ' + self.cluster_log_dir + '\n' \
               'cp $TMPDIR/${JOB_ID}_${SGE_TASK_ID}_stderr.txt ' + self.cluster_log_dir + '\n '

        logger.debug(f"Submit file: \n {text}")
        logger.debug(f"Creating file at {self.submit_file}")

        with open(self.submit_file, "w") as f:
            f.write(text)

        cmd = "chmod +x " + self.submit_file
        os.system(cmd)

    def submit_to_cluster(self, cluster_cpu, cluster_h, cluster_ram, tables, service, single_chunk=None):
        """
        Submit jobs to cluster

        :param cluster_cpu: Number of cluster CPUs
        :type cluster_cpu: int
        :param cluster_h: Time for cluster jobs
        :type cluster_h: str
        :param cluster_ram: RAM for cluster jobs
        :type cluster_ram: str
        :param tables: Table to query
        :type tables: str or list-like
        :param service: service to use for querying the data
        :type service: str
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

        ids = f'{_start_id}-{_end_id}'

        # make data_product files, storing essential info from parent_sample
        for jobID in range(_start_id, _end_id + 1):
            indices = np.where(self.cluster_jobID_map == jobID)[0]
            logger.debug(f"starting data_product for {len(indices)} objects.")
            data_product = self._start_data_product(parent_sample_indices=indices)
            chunk_number = self._get_chunk_number_for_job(jobID)
            self._save_data_product(data_product, service="tap", chunk_number=chunk_number, jobID=jobID)

        parentsample_class_pickle = os.path.join(self.cluster_dir, 'parentsample_class.pkl')
        logger.debug(f"pickling parent sample class to {parentsample_class_pickle}")
        with open(parentsample_class_pickle, "wb") as f:
            pickle.dump(self.parent_sample_class, f)

        submit_cmd = 'qsub '
        if cluster_cpu > 1:
            submit_cmd += "-pe multicore {0} -R y ".format(cluster_cpu)
        submit_cmd += f'-N wise_lightcurves '
        submit_cmd += f"-t {ids}:1 {self.submit_file}"
        logger.debug(f"Ram per core: {cluster_ram}")
        logger.info(f"{time.asctime(time.localtime())}: {submit_cmd}")

        self._make_cluster_script(cluster_h, cluster_ram, tables, service)

        try:
            msg = self._execute_bash_command(submit_cmd)
            logger.info(str(msg))
            job_id = int(str(msg).split('job-array')[1].split('.')[0])
            logger.info(f"Running on cluster with ID {job_id}")
            self.job_id = job_id
            return job_id

        except OSError:
            return

    def run_cluster(self, cluster_cpu, cluster_h, cluster_ram, service):
        """
        Run the DESY cluster

        :param cluster_cpu: Number of cluster CPUs
        :type cluster_cpu: int
        :param cluster_h: Time for cluster jobs
        :type cluster_h: str
        :param cluster_ram: RAM for cluster jobs
        :type cluster_ram: str
        :param service: service to use for querying the data
        :type service: str
        """

        self.clear_cluster_log_dir()
        self._save_cluster_info()
        self.submit_to_cluster(cluster_cpu, cluster_h, cluster_ram, tables=None, service=service)
        self.wait_for_job()
        for c in range(self.n_chunks):
            self._combine_data_products(service, chunk_number=c, remove=True, overwrite=True)

    # ---------------------------------------------------- #
    # END using cluster for downloading and binning        #
    # ----------------------------------------------------------------------------------- #

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
        data_product = self._load_data_product(service, chunk_number=_chunk_number)
        lc = pd.DataFrame.from_dict(data_product.loc[int(parent_sample_idx)]["timewise_lightcurve"])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--base_name', type=str)
    parser.add_argument('--min_sep_arcsec', type=float)
    parser.add_argument('--n_chunks', type=int)
    parser.add_argument('--tables', type=str, nargs='+', default='')
    parser.add_argument('--mag', type=bool, default=True)
    parser.add_argument('--flux', type=bool, default=False)
    parser.add_argument('--clear_unbinned', type=bool, default=False)
    parser.add_argument('--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    logging.getLogger("air_flares").setLevel(cfg.logging_level)
    logging.getLogger("timewise").setLevel(cfg.logging_level)

    wd = WISEDataDESYCluster(base_name=cfg.base_name,
                             min_sep_arcsec=cfg.min_sep_arcsec,
                             n_chunks=cfg.n_chunks,
                             parent_sample_class=None)
    wd._load_cluster_info()
    wd.clear_unbinned_photometry_when_binning = cfg.clear_unbinned
    chunk_number = wd._get_chunk_number_for_job(cfg.job_id)

    wd._subprocess_select_and_bin(service='tap', chunk_number=chunk_number, jobID=cfg.job_id)
    wd.calculate_metadata(service='tap', chunk_number=chunk_number, jobID=cfg.job_id)
