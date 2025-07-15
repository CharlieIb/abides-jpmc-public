from setuptools import setup, find_packages

setup(
    name='abides-multi-exchange',
    version='0.1.0',
    description='A multi-exchange simulation environment based on ABIDES.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'gym==0.18.0',
        'numpy',
        'abides-core', # Or whatever the correct name is on PyPI
        'abides-gym',
        'abides-markets',
    ],
    python_requires='>=3.7.8',
)
