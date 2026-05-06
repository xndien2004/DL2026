cd dien_workspace

python3 -m venv .dl

source .dl/bin/activate

cd DL2026

module purge

module load shared
module load rc-base
module load GCCcore/13.2.0
module load OpenBLAS/0.3.24-GCC-13.2.0
module load Python/3.11.5-GCCcore-13.2.0

pip install --upgrade pip
pip install -r requirements.txt

