import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QNN-Gen",
    version="0.0",
    author="Collin Farquhar",
    author_email="Farquhar13@gmail.com",
    description="A framework for quantum neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/farquhar13/QNN-Gen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
