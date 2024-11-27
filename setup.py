from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''Returns list of requirements'''
    with open(file_path, 'r') as f:
        print([line.strip() for line in f.readlines() if not line.startswith(('#', '-e .'))])
        return [line.strip() for line in f.readlines() if not line.startswith(('#', '-e .'))]



setup(
    name='ChurnPredictionTool',
    version='0.0.0',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
    author='Akhila Devarapalli',
    author_email='devarapalliakhila@gmail.com',
    description='Churn Prediction Tool',
    long_description='Churn Prediction Tool using Machine Learning'
    )