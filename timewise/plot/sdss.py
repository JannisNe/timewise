import requests
import os
import getpass
import logging
import matplotlib.pyplot as plt
import backoff

from ..util.backoff import backoff_hndlr


logger = logging.getLogger(__name__)


def get_sdss_credentials():
    if not os.environ.get("SDSS_USERID"):
        os.environ["SDSS_USERID"] = input("Enter SDSS user ID:")
    if not os.environ.get("SDSS_USERPW"):
        os.environ["SDSS_USERPW"] = getpass.getpass("Enter SDSS password:")
    return os.environ["SDSS_USERID"], os.environ["SDSS_USERPW"]


def login_to_sciserver():
    try:
        from SciServer import Authentication
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install SciServer (https://github.com/sciserver/SciScript-Python) "
            "if you want to see SDSS cutouts!"
        )

    uid, pw = get_sdss_credentials()
    logger.debug(f"logging in to SciServer with username {uid}")
    Authentication.login(uid, pw)


@backoff.on_exception(
    backoff.expo, requests.RequestException, max_tries=50, on_backoff=backoff_hndlr
)
def get_cutout(*args, **kwargs):
    login_to_sciserver()
    from SciServer import SkyServer

    return SkyServer.getJpegImgCutout(*args, **kwargs)


def plot_sdss_cutout(
    ra,
    dec,
    arcsec=20,
    arcsec_per_px=0.1,
    interactive=False,
    fn=None,
    title=None,
    save=False,
    ax=False,
    height=2.5,
):
    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    if not ax:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = plt.gcf()

    try:
        im = get_cutout(ra, dec, scale=arcsec_per_px, height=ang_px, width=ang_px)
        ax.imshow(
            im,
            origin="upper",
            extent=(
                (
                    arcsec / 2,
                    -arcsec / 2,
                    -arcsec / 2,
                    arcsec / 2,
                )
            ),
            cmap="gray",
        )
        ax.scatter(0, 0, marker="x", color="red")

    except Exception as e:
        if "outside the SDSS footprint" in str(e):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x = sum(xlim) / 2
            y = sum(ylim) / 2
            ax.annotate(
                "Outside SDSS Footprint",
                (x, y),
                color="red",
                ha="center",
                va="center",
                fontsize=20,
            )
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
