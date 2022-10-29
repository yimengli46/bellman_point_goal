from setuptools import setup, find_packages


setup(name='lsp_cond',
      version='1.0.0',
      description='Core code for Conditional Learned Subgoal Planning.',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
