import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='old_cutqc',
    version='0.0.1',
    description='CutQC Backend',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/weiT1993/CutQC.git',
    packages=setuptools.find_packages(),
    author='weiT1993',
    author_email='tangwei1027@gmail.com',
    license='MIT',
    python_requires='>=3.8',
    zip_safe=False)