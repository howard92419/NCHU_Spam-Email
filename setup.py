from setuptools import setup, find_packages

setup(
    name="nchu-spam-email",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "nltk>=3.6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    python_requires=">=3.8",
    author="NCHU",
    description="Spam Email Classification using Machine Learning",
    keywords="spam, email, classification, machine-learning, svm",
    project_urls={
        "Source": "https://github.com/howard92419/NCHU_Spam-Email",
    },
)