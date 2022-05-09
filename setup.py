from setuptools import find_packages, setup


# with open('requirements.txt') as f:
#     required = f.read().splitlines()


setup(
    name="HW1",
    packages=find_packages(),
    version="0.1.0",
    description="ML in production? homework 1",
    author="Uliana Bespalova",
    # entry_points={
    #     "console_scripts": [
    #         "ml_example_train = ml_example.train_pipeline:train_pipeline_command"
    #     ]
    # },
    # install_requires=required,
    license="MIT",
)