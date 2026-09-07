from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads requirements.txt and ignores flags like --extra-index-url or -e .
    """
    requirements = []
    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        for line in lines:
            req = line.strip()
            # Ignore empty lines, flags starting with '-', and '-e .'
            if req and not req.startswith("-") and req != HYPHEN_E_DOT:
                requirements.append(req)
    return requirements

setup(
    name="signature_recognition",
    version="0.0.1",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)