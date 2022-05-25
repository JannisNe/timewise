import requests, os, getpass, backoff
import pandas as pd
import matplotlib.pyplot as plt

from timewise.general import main_logger, cache_dir, backoff_hndlr
from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_by_visit import WiseDataByVisit


logger = main_logger.getChild(__name__)
mirong_url = 'http://staff.ustc.edu.cn/~jnac/data_public/wisevar.txt'
local_copy = os.path.join(cache_dir, 'mirong_sample.csv')


def get_mirong_sample():

    if not os.path.isfile(local_copy):

        logger.info(f'getting MIRONG sample from {mirong_url}')
        r = requests.get(mirong_url)
        lll = list()
        for l in r.text.split('\n')[1:]:
            illl = list()
            for ll in l.split(' '):
                if ll and '#' not in ll:
                    illl.append(ll)
            lll.append(illl)

        mirong_sample = pd.DataFrame(lll[1:-1], columns=lll[0])
        mirong_sample['ra'] = mirong_sample['RA']
        mirong_sample['dec'] = mirong_sample['DEC']
        logger.debug(f'saving to {local_copy}')

        mirong_sample.to_csv(local_copy, index=False)
        logger.info(f'found {len(mirong_sample)} objects in MIRONG Sample')
        mirong_sample.drop(columns=['ra', 'dec'], inplace=True)
        mirong_sample.to_csv(local_copy, index=False)

    else:
        logger.debug(f'loading {local_copy}')
        mirong_sample = pd.read_csv(local_copy)

    return mirong_sample


###########################################################################################################
#            START SDSS UTILS                       #
#####################################################


def get_sdss_credentials():
    if not os.environ.get('SDSS_USERID'):
        os.environ['SDSS_USERID'] = input('Enter SDSS user ID:')
    if not os.environ.get('SDSS_USERPW'):
        os.environ['SDSS_USERPW'] = getpass.getpass('Enter SDSS password:')
    return os.environ['SDSS_USERID'], os.environ['SDSS_USERPW']


def login_to_sciserver():
    try:
        from SciServer import Authentication
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install SciServer (https://github.com/sciserver/SciScript-Python) "
                                  "if you want to see SDSS cutouts!")

    uid, pw = get_sdss_credentials()
    logger.debug(f"logging in to SciServer with username {uid}")
    Authentication.login(uid, pw)


@backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=50,
        on_backoff=backoff_hndlr
    )
def get_cutout(*args, **kwargs):
    login_to_sciserver()
    from SciServer import SkyServer
    return SkyServer.getJpegImgCutout(*args, **kwargs)


def plot_sdss_cutout(ra, dec, arcsec=20, arcsec_per_px=0.1, interactive=False, fn=None, title=None, save=False, ax=False,
                height=2.5):

    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    if not ax:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = plt.gcf()

    try:
        im = get_cutout(ra, dec, scale=arcsec_per_px, height=ang_px, width=ang_px)
        ax.imshow(im, origin='upper',
                  extent=([ra + ang_deg / 2, ra - ang_deg / 2,
                           dec - ang_deg / 2, dec + ang_deg / 2]),
                  cmap='gray')
        ax.scatter(ra, dec, marker='x', color='red')

    except Exception as e:
        if "outside the SDSS footprint" in str(e):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x = sum(xlim) / 2
            y = sum(ylim) / 2
            ax.annotate("Outside SDSS Footprint", (x, y), color='red', ha='center', va='center', fontsize=20)
        else:
            raise Exception(e)

    if title:
        ax.set_title(title)

    if save:
        logger.debug(f"saving under {fn}")
        fig.savefig(fn)

    if interactive:
        return fig, ax

    plt.close()


#####################################################
#            END SDSS UTILS                         #
###########################################################################################################


###########################################################################################################
#            START EXCESS VARIANCE UTILS            #
#####################################################

# calculate excess variance as done in section 7.3.4 of (Boller et al)
# https://www.aanda.org/articles/aa/full_html/2016/04/aa25648-15/aa25648-15.html#S26


def calc_Expectation(a):
     
    # variable prb is for probability
    # of each element which is same for
    # each element
    n = len(a)
    prb = 1 / n
    # calculating expectation overall
    sum = 0
    for i in range(0, n):
        sum += (a[i] * prb)
    # returning expectation as sum
    return float(sum)


def get_excess_variance(y, y_err, mu):
    import numpy as np
    N = len(y)
    sum_variance = 0
    for i, (X, sig) in enumerate(zip(y, y_err)):
        sum_variance += np.power(X-mu,2) - np.power(sig,2)
       
    excess_variance = (sum_variance)/(N*mu**2)
    
    # calculate the uncertainty
    F_var = np.sqrt(np.abs(excess_variance))/mu
    std_exp = calc_Expectation(y_err**2)
    term1 = np.sqrt(2/N)*std_exp/(np.power(mu,2))
    term2 = np.sqrt((std_exp*2*F_var)/(N*mu))
    
    uncertainty = np.sqrt(term1**2 + term2**2)
    return excess_variance, uncertainty


#####################################################
#              END EXCESS VARIANCE UTILS            #
###########################################################################################################


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
            self.df = pd.DataFrame({'ra': [ra], 'dec': [dec], 'id': ['NGC 1068']})

        def _plot_cutout(self, ra, dec, arcsec, interactive, title=None, fn=None, save=False, ax=None, **kwargs):
            h = kwargs.get('height', 2)
            if not ax:
                fig, ax = plt.subplots(figsize=(h, h), sharex='all')
            else:
                fig = plt.gcf()
            plot_sdss_cutout(ra, dec, interactive=True, ax=ax, arcsec=arcsec, title='SDSS', **kwargs)

            if interactive:
                return fig, ax
            if save:
                fig.savefig(fn)
                plt.close()

            plt.show()
            plt.close()

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
    wd.get_photometric_data(**kwargs)
    wd.plot_lc(parent_sample_idx=0, service=kwargs.get('service', 'tap'))
    return wd


#####################################################
#            END POINT SOURCE  UTILS                #
###########################################################################################################
