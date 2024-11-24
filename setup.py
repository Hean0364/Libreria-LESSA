# setup.py

from setuptools import setup, find_packages

setup(
    name='lessa_lib',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    description='Librería para reconocimiento de lenguaje de señas de El Salvador',
    install_requires=[
        'opencv-python',
        'mediapipe',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'pyttsx3',
    ],
)
