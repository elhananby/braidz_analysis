#!/usr/bin/env python3
import gzip
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union


class BraidzError(Exception):
    """Base exception for BRAIDZ conversion errors."""

    pass


def get_output_path(input_dir: Union[str, Path]) -> Path:
    """
    Generate the output path for the BRAIDZ file based on the input directory.
    Appends 'z' to the input path.
    """
    input_path = Path(input_dir)
    input_str = str(input_path.resolve()).rstrip("/")
    return Path(f"{input_str}z")


def recompress_gzip(input_file: Path) -> Path:
    """
    Recompress a gzip file to ensure it's properly formatted.
    Creates a temporary file and returns its path.
    """
    print(f"Recompressing gzip file: {input_file}")
    temp_fd, temp_path = tempfile.mkstemp(suffix=".gz")
    os.close(temp_fd)
    temp_path = Path(temp_path)

    try:
        print("  Attempting to read as gzip...")
        with gzip.open(input_file, "rb") as f_in:
            with gzip.open(temp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("  Successfully recompressed as gzip")
    except EOFError:
        print("  Failed to read as gzip, attempting raw read...")
        # If we can't read it as gzip, try reading it as raw and then compress
        with open(input_file, "rb") as f_in:
            with gzip.open(temp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("  Successfully recompressed from raw data")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise BraidzError(f"Error recompressing {input_file}: {str(e)}")

    return temp_path


def dir_to_braidz(
    input_dirname: Union[str, Path], output_zipfile: Optional[Union[str, Path]] = None
) -> None:
    """
    Convert a directory to a BRAIDZ file (specialized ZIP format).
    """
    temp_files = []  # Keep track of temporary files to clean up

    try:
        input_dirname = Path(input_dirname)
        if output_zipfile is None:
            output_zipfile = get_output_path(input_dirname)
        else:
            output_zipfile = Path(output_zipfile)

        print(f"\nStarting conversion of {input_dirname} to {output_zipfile}")
        print("Creating BRAIDZ file with header...")

        # Create a new ZIP file and write the header
        with open(output_zipfile, "wb") as f:
            header = (
                "BRAIDZ file. This is a standard ZIP file with a "
                "specific schema. You can view the contents of this "
                "file at https://braidz.strawlab.org/\n"
            ).encode("utf-8")
            f.write(header)

        print("Scanning directory for files...")
        # Open the ZIP file in append mode to add files
        with zipfile.ZipFile(
            output_zipfile,
            mode="a",
            compression=zipfile.ZIP_STORED,  # No compression
            allowZip64=True,  # Support for large files
        ) as zipf:
            # Collect all files, separating README.md
            readme_entry = None
            other_files = []
            file_count = 0

            for root, _, files in os.walk(input_dirname):
                for file in files:
                    file_count += 1
                    full_path = Path(root) / file
                    if file == "README.md":
                        readme_entry = full_path
                        print(f"Found README.md: {full_path}")
                    else:
                        other_files.append(full_path)

            print(f"Found {file_count} files total")
            if readme_entry:
                print("README.md will be processed first")

            # Process files in the desired order (README.md first)
            files_to_process = ([readme_entry] if readme_entry else []) + other_files

            print("\nStarting file processing...")
            for i, file_path in enumerate(files_to_process, 1):
                if file_path is None:
                    continue

                # Calculate the relative path for the archive
                rel_path = file_path.relative_to(input_dirname)
                print(f"\nProcessing file {i}/{file_count}: {rel_path}")

                try:
                    # If it's a gzip file, recompress it to ensure proper format
                    if str(file_path).endswith(".gz"):
                        source_file = recompress_gzip(file_path)
                        temp_files.append(source_file)
                    else:
                        source_file = file_path
                        print("Adding file directly (not gzipped)")

                    # Set Unix permissions (0o755 = rwxr-xr-x)
                    zinfo = zipfile.ZipInfo(str(rel_path))
                    zinfo.external_attr = 0o755 << 16

                    # Add the file to the ZIP archive
                    file_size = os.path.getsize(source_file)
                    print(f"Adding to archive (size: {file_size / 1024 / 1024:.2f} MB)...")
                    with open(source_file, "rb") as f:
                        zipf.writestr(zinfo, f.read())
                    print("Successfully added to archive")

                except Exception as e:
                    raise BraidzError(f"Error adding {file_path} to archive: {str(e)}")

            print(f"\nAll {file_count} files processed successfully")

    except Exception as e:
        print("\nError occurred during processing!")
        # Clean up the output file if there was an error
        if output_zipfile.exists():
            print(f"Removing incomplete output file: {output_zipfile}")
            output_zipfile.unlink()
        raise BraidzError(f"Failed to create BRAIDZ file: {str(e)}")

    finally:
        if temp_files:
            print("\nCleaning up temporary files...")
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            print("Cleanup complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a directory to a BRAIDZ file")
    parser.add_argument("input_dir", type=str, help="Input directory to convert")
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        help="Output BRAIDZ file path (optional, defaults to input_dir + 'z')",
        default=None,
    )

    args = parser.parse_args()

    try:
        dir_to_braidz(args.input_dir, args.output_file)
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = get_output_path(args.input_dir)
        print(f"\nSuccessfully created BRAIDZ file: {output_path}")
    except BraidzError as e:
        logging.error(str(e))
        exit(1)
