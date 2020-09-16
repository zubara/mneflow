#! /usr/bin/env python
from setuptools import setup
import codecs
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name='mneflow',
        maintainer='Ivan Zubarev',
        maintainer_email='ivan.zubarev@aalto.fi',
        description='Interface for running tensorflow models on MEG/EEG data with MNE-python',
        license='BSD-3',
        url='https://github.com/zubara/mneflow',
        version='0.1.1-beta',
        download_url='https://github.com/zubara/mneflow/archive/master.zip',
        long_description=codecs.open('README.md', encoding='utf8').read(),
        long_description_content_type = 'text/markdown',
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        packages=['mneflow'],
        install_requires=['numpy', 'scipy', 'mne', 'tensorflow',  'matplotlib',
                          'sklearn', 'spectrum'],
    )
