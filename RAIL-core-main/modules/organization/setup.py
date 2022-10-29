from setuptools import setup, find_packages


setup(
    name="organization",
    version="1.0.0",
    description="Learned environment organization",
    license="MIT",
    author="Gregory J. Stein",
    author_email="gjstein@gmu.edu",
    packages=find_packages(),
    package_data={"": ["*.pddl"]},
    install_requires=["numpy", "matplotlib"],
)
