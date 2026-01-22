#!/usr/bin/env python3
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

from braidz_writer import dir_to_braidz
from tqdm import tqdm


class CleanupError(Exception):
    """Base exception for cleanup process errors."""

    pass


def is_valid_braidz(braidz_path: Path) -> bool:
    """
    Check if a .braidz file exists and is valid (not empty and can be opened).

    Args:
        braidz_path: Path to the .braidz file

    Returns:
        bool: True if the file is valid, False otherwise
    """
    try:
        if not braidz_path.exists():
            return False

        # Check if file is not empty (>1KB to account for header)
        if braidz_path.stat().st_size < 1024:
            return False

        # Try to open it as a zip file to verify integrity
        import zipfile

        with zipfile.ZipFile(braidz_path, "r") as zf:
            # Check if it has any files
            return len(zf.namelist()) > 0

    except Exception:
        return False

    return True


def find_braid_folders(base_dir: Union[str, Path]) -> List[Path]:
    """
    Find all .braid folders in the given directory.

    Args:
        base_dir: Base directory to search in

    Returns:
        List of paths to .braid folders
    """
    base_path = Path(base_dir)
    return [p for p in base_path.iterdir() if p.is_dir() and p.name.endswith(".braid")]


def process_braid_folder(braid_path: Path) -> Tuple[bool, str]:
    """
    Process a single .braid folder - either delete it if valid .braidz exists,
    or create new .braidz file.

    Args:
        braid_path: Path to the .braid folder

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        braidz_path = Path(str(braid_path) + "z")

        # Check if valid .braidz already exists
        if is_valid_braidz(braidz_path):
            # Remove the .braid folder
            shutil.rmtree(braid_path)
            return True, f"Deleted {braid_path} (valid .braidz exists)"

        # Create new .braidz file
        dir_to_braidz(braid_path)

        # Verify the new file
        if not is_valid_braidz(braidz_path):
            raise CleanupError("Created .braidz file is invalid")

        # Remove the original .braid folder
        shutil.rmtree(braid_path)
        return True, f"Created {braidz_path} and deleted {braid_path}"

    except Exception as e:
        return False, f"Error processing {braid_path}: {str(e)}"


def main(base_dir: Union[str, Path]) -> None:
    """
    Main function to process all .braid folders in a directory.

    Args:
        base_dir: Base directory containing .braid folders
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("braid_cleanup.log")],
    )

    try:
        # Find all .braid folders
        braid_folders = find_braid_folders(base_dir)
        if not braid_folders:
            print(f"No .braid folders found in {base_dir}")
            return

        print(f"Found {len(braid_folders)} .braid folders")

        # Process each folder with progress bar
        successes = []
        failures = []

        with tqdm(total=len(braid_folders), desc="Processing folders") as pbar:
            for folder in braid_folders:
                success, message = process_braid_folder(folder)
                if success:
                    successes.append((folder, message))
                else:
                    failures.append((folder, message))
                logging.info(message)
                pbar.update(1)

        # Print summary
        print("\nProcessing complete!")
        print(f"Successfully processed: {len(successes)} folders")
        print(f"Failed to process: {len(failures)} folders")

        if failures:
            print("\nFailed folders:")
            for folder, message in failures:
                print(f"  {folder}: {message}")

        print(f"\nFull details available in: {os.path.abspath('braid_cleanup.log')}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process .braid folders in a directory, converting to .braidz and cleaning up"
    )
    parser.add_argument("base_dir", type=str, help="Base directory containing .braid folders")

    args = parser.parse_args()

    try:
        main(args.base_dir)
    except Exception as e:
        logging.error(str(e))
        exit(1)
