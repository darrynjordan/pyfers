from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyfers",
    version='1.1.1',
    author="Darryn Anton Jordan",
    author_email="<darrynjordan@icloud.com>",
    description='XML generator for FERS',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'h5py'],
    keywords=['radar', 'simulation', 'fers', 'uct'],
    url="https://github.com/darrynjordan/FERS",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
