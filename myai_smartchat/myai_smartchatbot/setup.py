from setuptools import setup, find_packages

setup(
    name="myai_smartchat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "duckduckgo-search",
    ],
    python_requires=">=3.9",
)
