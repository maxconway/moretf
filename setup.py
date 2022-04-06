from setuptools import setup

setup(
name='moretf',
version='0.0.1',
author='Max Conway',
author_email='conway.max1@gmail.com',
packages=['moretf', 'moretf.test'],
# scripts=['bin/script1','bin/script2'],
# url='http://pypi.python.org/pypi/PackageName/',
license='LICENSE.txt',
# description='An awesome package that does something',
long_description=open('README.md').read(),
install_requires=[
    "tensorflow >= 2.6.2",
    "pytest",
],
)