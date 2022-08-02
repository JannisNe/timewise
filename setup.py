import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setuptools.setup(
        name="timewise",
        version="0.1.8",
        author="Jannis Necker",
        author_email="jannis.necker@gmail.com",
        description="A small package to download infrared data from the WISE satellite",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="MIT",
        keywords="WISE data infrared astronomy",
        url="https://github.com/JannisNe/timewise",
        project_urls={
            "Bug Tracker": "https://github.com/JannisNe/timewise/issues",
        },
        packages=setuptools.find_packages(),
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        python_requires='>=3.8',
        install_requires=[
            "tqdm==4.64.0",
            "requests==2.28.1",
            "pandas==1.4.3",
            "numpy==1.23.1",
            "pyvo==1.3",
            "astropy==5.1",
            "matplotlib==3.5.2",
            "coveralls==3.3.1",
            "scikit-image==0.19.3",
            "backoff==2.1.2",
            "myst-parser==0.18.0"
        ],
        package_data={'timewise': [
            'wise_flux_conversion_correction.dat'
        ]}
    )