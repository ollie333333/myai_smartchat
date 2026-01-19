from setuptools import setup, find_packages

setup(
    name="myai_smartchat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ddgs"
    ],
    python_requires=">=3.9",
    description="Smart Python chatbot with DuckDuckGo search",
    author="Ollie",
)
