import requests, os, getpass
import pandas as pd
import matplotlib.pyplot as plt

try:
    from SciServer import SkyServer, Authentication
except ModuleNotFoundError:
    SkyServer = Authentication = None

from timewise.general import main_logger, cache_dir


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

    if isinstance(SkyServer, type(None)) or isinstance(Authentication, type(None)):
        raise ModuleNotFoundError("Please install SciServer (https://github.com/sciserver/SciScript-Python) "
                                  "if you want to see SDSS cutouts!")

    uid, pw = get_sdss_credentials()
    logger.debug(f"logging in to SciServer with username {uid}")
    Authentication.login(uid, pw)


def plot_sdss_cutout(ra, dec, arcsec=20, arcsec_per_px=0.1, interactive=False, fn=None, title=None, save=False, ax=False,
                height=2.5):

    login_to_sciserver()

    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    if not ax:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = plt.gcf()

    try:
        im = SkyServer.getJpegImgCutout(ra, dec, scale=arcsec_per_px, height=ang_px, width=ang_px)
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
        fig.savefig(fn)

    if interactive:
        return fig, ax

    plt.close()

#####################################################
#            END SDSS UTILS                         #
###########################################################################################################
