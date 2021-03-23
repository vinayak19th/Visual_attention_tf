import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visual-attention-tf", # Replace with your own username
    version="1.1.0",
    author="Vinayak Sharma",
    author_email="vinayak19th@gmail.com",
    description="CNN Attention layer to be used with tf or tf.keras ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinayak19th/Visual_attention_tf",
    project_urls={
        "Bug Tracker": "https://github.com/vinayak19th/Visual_attention_tf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires = ["tensorflow>=2.2.0"],
    license='MIT',
)