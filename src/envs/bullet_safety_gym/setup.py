from setuptools import setup

setup(
    name='bullet-safety-gym',
    version='0.1.0',
    description='Open Safety Gym environments based on PyBullet',
    # setup.py lives inside the bullet_safety_gym/ directory, so the package
    # root is "." and the envs sub-package is "./envs".
    packages=['bullet_safety_gym', 'bullet_safety_gym.envs'],
    package_dir={
        'bullet_safety_gym': '.',
        'bullet_safety_gym.envs': 'envs',
    },
    package_data={
        'bullet_safety_gym': ['envs/data/**/*'],
    },
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
    ],
)
