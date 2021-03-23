import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="visual-attention-tf",
    version="1.0.0",
    description="CNN Attention layer to be used with tf or tf.keras",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vinayak19th/Visual_attention_tf",
    author="Vinayak Sharma",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
    ],
    packages=["visual-attention-tf"],
    include_package_data=True,
    install_requires=["tensorflow"]
)