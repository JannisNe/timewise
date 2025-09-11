import logging
import backoff
import pyvo as vo
from xml.etree import ElementTree


logger = logging.getLogger(__name__)


def backoff_hndlr(details):
    logger.info(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )


class StableAsyncTAPJob(vo.dal.AsyncTAPJob):
    """
    Implements backoff for call of phase which otherwise breaks the code if there are connection issues.
    Also stores the response of TapQuery.submit() under self.submit_response for debugging
    """

    def __init__(self, url, *, session=None, delete=True):
        super(StableAsyncTAPJob, self).__init__(url, session=session, delete=delete)
        self.submit_response = None

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
        tapquery = vo.dal.TAPQuery(
            baseurl,
            query,
            mode="async",
            language=language,
            maxrec=maxrec,
            uploads=uploads,
            session=session,
            **keywords,
        )
        response = tapquery.submit()
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        info = root.find(".//v:INFO", {"v": "http://www.ivoa.net/xml/VOTable/v1.3"})
        if info.attrib.get("value") == "ERROR":
            raise vo.dal.DALQueryError(info.text.strip())
        job = cls(response.url, session=session)
        job._client_set_maxrec = maxrec
        job.submit_response = response
        return job

    @property
    @backoff.on_exception(
        backoff.expo,
        (vo.dal.DALServiceError, AttributeError),
        max_tries=50,
        on_backoff=backoff_hndlr,
    )
    def phase(self):
        return super(StableAsyncTAPJob, self).phase


class StableTAPService(vo.dal.TAPService):
    """
    Implements the StableAsyncTAPJob for job submission
    """

    @backoff.on_exception(
        backoff.expo,
        (vo.dal.DALServiceError, AttributeError, AssertionError),
        max_tries=5,
        on_backoff=backoff_hndlr,
    )
    def submit_job(
        self, query, *, language="ADQL", maxrec=None, uploads=None, **keywords
    ):
        job = StableAsyncTAPJob.create(
            self.baseurl,
            query,
            language=language,
            maxrec=maxrec,
            uploads=uploads,
            session=self._session,
            **keywords,
        )
        logger.debug(job.url)
        assert job.phase
        return job
