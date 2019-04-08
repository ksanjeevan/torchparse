from setuptools import setup

setup(
    name="torchparse",
    version='0.1',
    author='Kiran Sanjeevan',
    url='http://github.com/ksanjeevan/torchparse',
    author_email="ksanjeevancabeza@gmail.com",
    description="PyTorch Model Parser: Parse and create neural nets easily defined in a .cfg file",
    packages=['torchparse'],
    license='MIT',
    install_requires=[
        'numpy >= 1.16.2',
        'torch >= 1.0'
      ],
    keywords="torch deep learning parse cfg",
    zip_safe=False
)
