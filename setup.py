from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "semops" / "README.md"


setup(
    name="semops",
    version="0.2.0",
    description="Semantic operations for Python. The NumPy of meaning.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={
        "openai": ["openai>=1.0"],
        "local": ["sentence-transformers>=2.0"],
        "parquet": ["pyarrow>=15.0"],
    },
)
