import logging
import hashlib
from pathlib import Path
from io import BytesIO
from PIL import Image

import requests
import matplotlib.pyplot as plt
import numpy
from astropy.table import Table
import astropy.config.paths


logger = logging.getLogger(__name__)


class PanSTARRSQueryError(Exception):
    pass


def load_cache_or_download(url):
    logger.debug(f"loading or downloading {url}")
    h = hashlib.md5(url.encode()).hexdigest()
    cache_dir = Path(astropy.config.paths.get_cache_dir())
    cache_file = cache_dir / (h + ".cache")
    logger.debug(f"cache file is {cache_file}")
    if not cache_file.is_file():
        logger.debug(f"downloading {url}")
        r = requests.get(url)
        with open(cache_file, "wb") as f:
            f.write(r.content)
        return r.content
    else:
        logger.debug(f"loading {cache_file}")
        with open(cache_file, "rb") as f:
            return f.read()


def annotate_not_available(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = sum(xlim) / 2
    y = sum(ylim) / 2
    logger.debug(f"annotate_not_available at {x}, {y}")
    ax.annotate(
        "Outside\nPanSTARRS\nFootprint",
        (x, y),
        color="red",
        ha="center",
        va="center",
        fontsize=10,
    )


def getimages(ra, dec, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    content = load_cache_or_download(url)
    table = Table.read(content.decode(), format="ascii")
    return table


def geturl(
    ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False
):
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
    if len(table) == 0:
        raise PanSTARRSQueryError("No images available")
    url = (
        f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size}&format={format}"
    )
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table["filter"]]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table["filename"][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
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
    url = geturl(
        ra,
        dec,
        size=size,
        filters=filters,
        output_size=output_size,
        format=format,
        color=True,
    )
    content = load_cache_or_download(url)
    im = Image.open(BytesIO(content))
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
    url = geturl(
        ra, dec, size=size, filters=filter, output_size=output_size, format=format
    )
    content = load_cache_or_download(url[0])
    im = Image.open(BytesIO(content))
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
    height=2.5,
):
    arcsec_per_px = 0.25
    ang_px = int(arcsec / arcsec_per_px)

    imshow_kwargs = {
        "origin": "upper",
        "extent": ([arcsec / 2, -arcsec / 2, -arcsec / 2, arcsec / 2]),
    }
    scatter_args = [0, 0]
    scatter_kwargs = {"marker": "x", "color": "red"}

    if not plot_color_image:
        filters = "grizy"
        if not ax:
            fig, axss = plt.subplots(
                2,
                len(filters),
                sharex="all",
                sharey="all",
                gridspec_kw={"wspace": 0, "hspace": 0, "height_ratios": [1, 8]},
                figsize=(height * 5, height),
            )
        else:
            fig = plt.gcf()
            axss = ax

        for j, fil in enumerate(list(filters)):
            axs = axss[1]
            try:
                im = getgrayim(ra, dec, size=ang_px, filter=fil)
                axs[j].imshow(im, cmap="gray", **imshow_kwargs)
            except PanSTARRSQueryError:
                axs[j].set_xlim(-arcsec / 2, arcsec / 2)
                axs[j].set_ylim(-arcsec / 2, arcsec / 2)
                annotate_not_available(axs[j])

            axs[j].scatter(*scatter_args, **scatter_kwargs)
            axs[j].set_title(fil)
            axss[0][j].axis("off")

    else:
        logger.debug("plotting color image")
        if not ax:
            fig, axss = plt.subplots(figsize=(height, height))
        else:
            fig = plt.gcf()
            axss = ax

        try:
            im = getcolorim(ra, dec, size=ang_px)
            axss.imshow(im, **imshow_kwargs)
        except PanSTARRSQueryError:
            axss.set_xlim(-arcsec / 2, arcsec / 2)
            axss.set_ylim(-arcsec / 2, arcsec / 2)
            annotate_not_available(axss)
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
        logger.info(f"saving under {fn}")
        fig.savefig(fn)

    if interactive:
        return fig, axss

    plt.close()
