# setup.py
from setuptools import setup, find_packages

setup(
    name="gelsight_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rclpy",
        "ur-rtde",
        "opencv-python",
        # …and any other pip deps
    ],
    entry_points={
        "console_scripts": [
            "gelsight-controller=controller_node:main",
        ],
    },
)
