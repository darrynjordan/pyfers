from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyfers",
    version='2.0.6',
    author="Darryn Anton Jordan",
    author_email="<darrynjordan@icloud.com>",
    description='Generates XML Descriptors for FERS',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'lxml', 'scipy'],
    keywords=['radar', 'simulation'],
    url="https://github.com/darrynjordan/pyfers",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
