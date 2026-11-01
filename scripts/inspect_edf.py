import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.eeg_loader import inspect_eeg_file

def main():
    data_dir = PROJECT_ROOT / "data" / "raw"
    edf_files = sorted(list(data_dir.glob("*.edf")) + list(data_dir.glob("*.bdf")))

    if not edf_files:
        print(f"No edf files found in {data_dir.resolve()}")
        return

    print(f"Found {len(edf_files)} recording(s) in {data_dir}:")
    print("-" * 75)
    for f in edf_files:
        info = inspect_eeg_file(f)
        print(f"File:        {info['filename']}")
        print(f"Channels:    {info['n_channels']}")
        print(f"Sampling:    {info['sfreq']} Hz")
        print(f"Samples:     {info['n_samples']}")
        print(f"Duration:    {info['duration_sec']:.2f} seconds ({info['duration_sec']/60:.2f} min)")
        print("-" * 75)

if __name__ == "__main__":
    main()