import pandas as pd
import pyvo as vo
import time
import logging
from pathlib import Path
from astropy.table import Table

from tests.test_queries import normalize_sql


logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"


def get_table_from_query_and_chunk(query: str, chunk: int | str) -> Table:
    normalized_queries = {}
    for t in ["allwise_p3as_mep", "neowiser_p1bs_psd"]:
        fn = DATA_DIR / "queries" / f"positional_{t}_mag_fluxes.txt"
        logger.debug(f"reading {fn}")
        normalized_queries[normalize_sql(fn.read_text())] = t
    t = normalized_queries[normalize_sql(query)]

    fn = DATA_DIR / "photometry" / f"raw_photometry_{t}__chunk{chunk}.csv"
    logger.debug(f"loading {fn}")
    return Table.from_pandas(pd.read_csv(fn, index_col=0))


class DummyAsyncTAPJob:
    """
    Hacky drop-in replacement for AsyncTAPJob
    """

    def __init__(
        self,
        url,
        *,
        session=None,
        delete=True,
        fail_fetch=False,
        final_phase="COMPLETED",
    ):
        self.url = url
        self.fail_fetch = fail_fetch
        self.final_phase = final_phase

    def _update(self, wait_for_statechange=False, timeout=10.0):
        pass

    def run(self):
        pass

    @classmethod
    def create(
        cls,
        baseurl,
        query,
        *,
        language="ADQL",
        maxrec=None,
        uploads=None,
        session=None,
        chunksize=None,
        **keywords,
    ):
        """
        creates a async tap job on the server under ``baseurl``
        Raises requests.HTTPError if TAPQuery.submit() failes.

        Parameters
        ----------
        baseurl : str
            the TAP baseurl
        query : str
            the query string
        language : str
            specifies the query language, default ADQL.
            useful for services which allow to use the backend query language.
        maxrec : int
            the maximum records to return. defaults to the service default
        uploads : dict
            a mapping from table names to objects containing a votable
        session : object
           optional session to use for network requests
        """
        chunk_id = uploads["mine"]["orig_id"].max() // chunksize
        logger.debug(f"chunk id: {chunk_id}")
        url = query + f"_chunk{chunk_id}_created{time.time()}"
        # create the job instance
        job = cls(url, session=session)
        job._client_set_maxrec = maxrec
        job.submit_response = ""
        return job

    @property
    def phase(self):
        created = float(self.url.split("created")[-1])
        if time.time() - created > 1:
            return self.final_phase
        return "RUNNING"

    def fetch_result(self):
        if self.fail_fetch:
            raise vo.DALServiceError("failed fetch")
        q = normalize_sql(self.url.split("_chunk")[0])
        c = self.url.split("chunk")[1][0]
        t = get_table_from_query_and_chunk(q, c)

        def to_table():
            return t

        t.to_table = to_table
        return t

    def wait(self):
        pass


class DummyTAPService(vo.dal.TAPService):
    """
    A hacky drop-in replacement for TAPService to be used in testing
    """

    def __init__(
        self,
        baseurl,
        chunksize,
        *,
        capability_description=None,
        session=None,
        fail_submit=False,
        fail_fetch=False,
        final_job_phase="COMPLETED",
        sync_res: Table | None = None,
    ):
        super(DummyTAPService, self).__init__(
            baseurl, capability_description=None, session=None
        )
        self.chunksize = chunksize
        self.fail_submit = fail_submit
        self.fail_fetch = fail_fetch
        self.sync_res = sync_res
        self.final_job_phase = final_job_phase

    def submit_job(
        self, query, *, language="ADQL", maxrec=None, uploads=None, **keywords
    ):
        if self.fail_submit:
            raise vo.dal.exceptions.DALServiceError("failed submit")

        job = DummyAsyncTAPJob.create(
            self.baseurl,
            query,
            language=language,
            maxrec=maxrec,
            uploads=uploads,
            session=self._session,
            chunksize=self.chunksize,
            final_phase=self.final_job_phase,
            **keywords,
        )
        logger.debug(job.url)
        assert job.phase
        return job

    def get_job_from_url(self, url):
        return DummyAsyncTAPJob(
            url=url,
            session=self._session,
            fail_fetch=self.fail_fetch,
            final_phase=self.final_job_phase,
        )

    def run_sync(
        self, query, *, language="ADQL", maxrec=None, uploads=None, **keywords
    ):
        class DummyTAPResult:
            def to_table(self_inner):
                return self.sync_res

        return DummyTAPResult()
