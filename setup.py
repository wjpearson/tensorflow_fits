import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-fits",
    version="1.0.0",
    author="Willaim J. Pearson",
    author_email="willjamespearson@gmail.com",
    license="Apache-2.0",
    description="Load FITS files into tf.data.Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wjpearson/tensorflow_fits",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
                ],
    python_requires='>=3.5',
    install_requires=['tensorflow>=2.0.0a0'],
)
