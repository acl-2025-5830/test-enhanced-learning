from setuptools import setup, find_packages

setup(
    name='TestEnhancedLearning',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "transformers",
        "torch==2.4.1",
        "datasets",
        "numpy<2.0.0",
        "ipykernel",
        "ipywidgets",
        "jsonlines",
        "bitsandbytes",
        "accelerate",
        "peft",
        "seaborn"
    ],
)