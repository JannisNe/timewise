import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setuptools.setup(
        name="timewise",
        version="0.1.7",
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
            "tqdm==4.63.0",
            "requests==2.27.1",
            "pandas==1.4.1",
            "numpy==1.22.2",
            "pyvo==1.3",
            "astropy==5.0",
            "matplotlib==3.5.1",
            "coveralls==3.3.1",
            "scikit-image==0.19.2",
        ],
        package_data={'timewise': [
            'wise_flux_conversion_correction.dat'
        ]}
    )