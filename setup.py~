
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='shapeGMM',
    version='0.0.5',
    author='Martin McCullagh',
    author_email='martin.mccullagh@okstate.edu',
    description='Gaussian Mixture Model clustering in size-and-shape space',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mccullaghlab/GMM-Positions',
    project_urls = {
        "Bug Tracker": "https://github.com/mccullaghlab/GMM-Positions/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    license='MIT',
    install_requires=['numpy','numba','sklearn'],
)
