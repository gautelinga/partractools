from importlib_metadata import entry_points
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="partractools",
    version="0.0.1",
    author="Gaute Linga",
    author_email="gaute.linga@mn.uio.no",
    description="Tools for partrac",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gautelinga/partractools",
    project_urls={
        "Bug Tracker": "https://github.com/gautelinga/partractools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"partractools": "partractools"},
    packages=["partractools",
              "partractools.common",
              "partractools.felbm",
              "partractools.fenics"],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["make_xdmf=partractools.make_xdmf:main",
                                      "analyze_elongation=partractools.analyze_elongation:main",
                                      "analyze_filaments=partractools.analyze_filaments:main",
                                      "measure_exponents=partractools.measure_exponents:main"]},
)