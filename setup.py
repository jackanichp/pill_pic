from setuptools import find_packages
from setuptools import setup

with open("requirements_prod.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='pill_pic',
      version="0.0.12",
      description="Pill Pic Model (api_pred)",
      license="MIT",
      author="Paul",
      author_email="jackanichp@gmail.com",
      #url="https://github.com/jackanichp/pill_pic",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
