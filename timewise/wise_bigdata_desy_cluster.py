import getpass, os, time, subprocess, math, pickle, queue, threading, argparse
import numpy as np
import pandas as pd

from timewise.general import main_logger, DATA_DIR_KEY, data_dir
from timewise.wise_data_by_visit import WiseDataByVisit


logger = main_logger.getChild(__name__)


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

        self._tap_queue = queue.Queue()
        self._cluster_queue = queue.Queue()

    def get_sample_photometric_data(self, max_nTAPjobs=8, perc=1, tables=None, chunks=None,
                                    cluster_jobs_per_chunk=100, wait=5, remove_chunks=True, query_type='positional'):
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
        :param remove_chunks: remove single chink files after binning
        :type remove_chunks: bool
        """

        mag = True
        flux = True

        if tables is None:
            tables = [
                'AllWISE Multiepoch Photometry Table',
                'NEOWISE-R Single Exposure (L1b) Source Table'
            ]

        if chunks is None:
            chunks = list(range(round(int(self.n_chunks * perc))))

        if query_type not in self.query_types:
            raise ValueError(f"Unknown query type {query_type}! Choose one of {self.query_types}")

        service = 'tap'
        self.queue = queue.Queue()
        tables = np.atleast_1d(tables)
        self.tap_jobs = {t: dict() for t in tables}

        logger.debug(f"Getting {perc * 100:.2f}% of lightcurve chunks ({len(chunks)}) via {service} "
                     f"in {'magnitude' if mag else ''} {'flux' if flux else ''} "
                     f"from {tables}")

        input('Correct? [hit enter] ')

        self.n_cluster_jobs_per_chunk = cluster_jobs_per_chunk
        cluster_time_s = max(len(self.parent_sample.df) / self._n_chunks / self.n_cluster_jobs_per_chunk, 59 * 60)
        if cluster_time_s > 24 * 3600:
            raise ValueError(f"cluster time per job would be longer than 24h! "
                             f"Choose more than {self.n_cluster_jobs_per_chunk} jobs per chunk!")

        cluster_time = time.strftime('%H:%M:%S', time.gmtime(cluster_time_s))
        self.clear_cluster_log_dir()
        self._save_cluster_info()

        tap_threads = [threading.Thread(target=self._tap_thread, daemon=True) for _ in range(max_nTAPjobs)]
        cluster_threads = [threading.Thread(target=self._cluster_thread, daemon=True) for _ in range(max_nTAPjobs)]

        for t in tap_threads + cluster_threads:
            logger.debug('starting thread')
            t.start()

        logger.debug(f'started {len(tap_threads)} TAP threads and {len(cluster_threads)} cluster threads.')

        for c in chunks:
            self._tap_queue.put((tables, c, wait, mag, flux, cluster_time, query_type))

        logger.debug(f'added {self._tap_queue.qsize()} tasks to tap queue')
        self._tap_queue.join()
        logger.debug('TAP done')
        self._cluster_queue.join()
        logger.debug('cluster done')

        self._combine_lcs(service=service, overwrite=True, remove=remove_chunks)
        self._combine_metadata(service=service, overwrite=True, remove=remove_chunks)

    def _tap_thread(self):
        logger.debug('started tap thread')
        while True:
            tables, chunk, wait, mag, flux, cluster_time, query_type = self._tap_queue.get(block=True)
            logger.debug(f'querying IRSA for chunk {chunk}')

            for t in tables:
                self._submit_job_to_TAP(chunk, t, mag, flux, query_type)
                logger.info(f'wating for {wait} hours')
                time.sleep(wait * 3600)
                self._thread_wait_and_get_results(t, chunk)

            logger.info(f'got all TAP results for chunk {chunk}. submitting to cluster')
            job_id = self.submit_to_cluster(cluster_cpu=1,
                                            cluster_h=cluster_time,
                                            cluster_ram='10G',
                                            tables=None,
                                            service='tap',
                                            single_chunk=chunk)

            self._tap_queue.task_done()
            self._cluster_queue.put((job_id, chunk))

    def _cluster_thread(self):
        logger.debug('started cluster thread')
        while True:
            job_id, chunk = self._cluster_queue.get(block=True)
            logger.debug(f'waiting for chunk {chunk} (Cluster job {job_id})')
            self.wait_for_job(job_id)
            logger.debug(f'cluster done for chunk {chunk} (Cluster job {job_id}). Start combining')
            try:
                self._combine_lcs('tap', chunk_number=chunk, remove=True, overwrite=True)
                self._combine_metadata('tap', chunk_number=chunk, remove=True, overwrite=True)
            finally:
                self._cluster_queue.task_done()

    # ----------------------------------------------------------------------------------- #
    # START using cluster for downloading and binning      #
    # ---------------------------------------------------- #

    @staticmethod
    def _qstat_output(qstat_command):
        """return the output of the qstat_command"""
        # start a subprocess to query the cluster
        with subprocess.Popen(qstat_command, stdout=subprocess.PIPE, shell=True) as process:
            # read the output
            tmp = process.stdout.read().decode()
            process.terminate()
            msg = str(tmp)
        return msg

    @staticmethod
    def _get_ids(qstat_command):
        """Takes a command that queries the DESY cluster and returns a list of job IDs"""
        st = WISEDataDESYCluster._qstat_output(qstat_command)
        # If the output is an empty string there are no tasks left
        if st == '':
            ids = list()
        else:
            # Extract the list of job IDs
            ids = np.array([int(s.split(' ')[2]) for s in st.split('\n')[2:-1]])
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
                if i > 3:
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
            ids = f'1-{self.n_chunks*self.n_cluster_jobs_per_chunk}'
        else:
            _start_id = int(single_chunk*self.n_cluster_jobs_per_chunk) + 1
            _end_id = int(_start_id + self.n_cluster_jobs_per_chunk)
            ids = f'{_start_id}-{_end_id}'

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

        with subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True) as process:
            msg = process.stdout.read().decode()
            process.terminate()

        logger.info(str(msg))
        job_id = int(str(msg).split('job-array')[1].split('.')[0])
        logger.info(f"Running on cluster with ID {job_id}")
        self.job_id = job_id
        return job_id

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
            self._combine_lcs(service, chunk_number=c, remove=True, overwrite=True)
            self._combine_metadata(service, chunk_number=c, remove=True, overwrite=True)

    # ---------------------------------------------------- #
    # END using cluster for downloading and binning        #
    # ----------------------------------------------------------------------------------- #


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

    main_logger.setLevel(cfg.logging_level)

    wd = WISEDataDESYCluster(base_name=cfg.base_name,
                             min_sep_arcsec=cfg.min_sep_arcsec,
                             n_chunks=cfg.n_chunks,
                             parent_sample_class=None)
    wd._load_cluster_info()
    wd.clear_unbinned_photometry_when_binning = cfg.clear_unbinned
    chunk_number = wd._get_chunk_number_for_job(cfg.job_id)

    wd._subprocess_select_and_bin(chunk_number=chunk_number, jobID=cfg.job_id)
    wd.calculate_metadata(service='tap', chunk_number=chunk_number, jobID=cfg.job_id)