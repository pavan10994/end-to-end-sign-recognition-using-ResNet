from typing import List
from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Parses requirements.txt and returns a list of dependencies,
    excluding editable package installation tags.
    """
    requirements = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "").strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="signature_recognition",
    version="0.0.1",
    author="MLOps Team",
    author_email="mlops@signature.ai",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="Production-Grade Offline Signature Recognition Pipeline using PyTorch, FastAPI, and GCP",
    python_requires=">=3.9",
)