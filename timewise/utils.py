import requests
import os
import getpass
import logging
import pandas as pd
import matplotlib.pyplot as plt
import pyvo as vo
import backoff
import numpy
from astropy.table import Table
from PIL import Image
from io import BytesIO


from timewise.general import cache_dir, backoff_hndlr


logger = logging.getLogger(__name__)
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
        requests.RequestException,
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
#            START PANSTARRS UTILS                  #
#####################################################


def getimages(ra, dec, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra, dec, size=size, filters=filters, output_size=output_size, format=format, color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra, dec, size=size, filters=filter, output_size=output_size, format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im


def plot_panstarrs_cutout(
        ra,
        dec,
        arcsec,
        interactive=False,
        fn=None,
        title=None,
        save=False,
        ax=False,
        plot_color_image=False,
        height=2.5
):
    arcsec_per_px = 0.25
    ang_px = int(arcsec / arcsec_per_px)

    imshow_kwargs = {
        'origin': 'upper',
        "extent": ([arcsec / 2, -arcsec / 2, -arcsec / 2, arcsec / 2])
    }
    scatter_args = [0, 0]
    scatter_kwargs = {'marker': 'x', 'color': 'red'}

    if not plot_color_image:
        filters = 'grizy'
        if not ax:
            fig, axss = plt.subplots(2, len(filters), sharex='all', sharey='all',
                                     gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [1, 8]},
                                     figsize=(height * 5, height))
        else:
            fig = plt.gcf()
            axss = ax

        for j, fil in enumerate(list(filters)):
            im = getgrayim(ra, dec, size=ang_px, filter=fil)
            axs = axss[1]
            axs[j].imshow(im, cmap='gray', **imshow_kwargs)

            axs[j].scatter(*scatter_args, **scatter_kwargs)
            axs[j].set_title(fil)
            axss[0][j].axis('off')

    else:
        logger.debug('plotting color image')
        if not ax:
            fig, axss = plt.subplots(figsize=(height, height))
        else:
            fig = plt.gcf()
            axss = ax

        im = getcolorim(ra, dec, size=ang_px)
        axss.imshow(im, **imshow_kwargs)
        axss.scatter(*scatter_args, **scatter_kwargs)

    _this_title = title if title else f"{ra}_{dec}"
    si = "-" if dec > 0 else "+"
    ylabel = f"Dec {si} {abs(dec):.2f} [arcsec]"
    xlabel = f"RA - {ra:.2f} [arcsec]"
    try:
        axss.set_title(_this_title)
        axss.set_xlabel(xlabel)
        axss.set_ylabel(ylabel)
        axss.grid(ls=":", alpha=0.5)
    except AttributeError:  # in this case axss is an array
        fig.supylabel(ylabel)
        fig.supxlabel(xlabel)
        fig.suptitle(_this_title)
        for a in axss.flatten():
            a.grid(ls=":", alpha=0.5)

    if save:
        logger.info(f'saving under {fn}')
        fig.savefig(fn)

    if interactive:
        return fig, axss

    plt.close()


#####################################################
#            END PANSTARRS UTILS                    #
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
#            START CUSTOM TAP Service                 #
#######################################################


class StableAsyncTAPJob(vo.dal.AsyncTAPJob):
    """
    Implements backoff for call of phase which otherwise breaks the code if there are connection issues
    """

    @property
    @backoff.on_exception(
        backoff.expo,
        (vo.dal.DALServiceError, AttributeError),
        max_tries=50,
        on_backoff=backoff_hndlr
    )
    def phase(self):
        return super(StableAsyncTAPJob, self).phase


class StableTAPService(vo.dal.TAPService):
    """
    Implements the StableAsyncTAPJob for job submission
    """

    def submit_job(
            self,
            query,
            language="ADQL",
            maxrec=None,
            uploads=None,
            **keywords
    ):
        return StableAsyncTAPJob.create(
            self.baseurl,
            query,
            language,
            maxrec,
            uploads,
            self._session,
            **keywords
        )


#######################################################
#            END CUSTOM TAP Service                   #
###########################################################################################################
