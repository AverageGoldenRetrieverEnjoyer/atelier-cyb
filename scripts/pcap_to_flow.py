#!/usr/bin/env python3
"""
Wrapper script for CICFlowMeter Docker container.
Usage: python pcap_to_flow.py input.pcap output.csv
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python pcap_to_flow.py input.pcap output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Create tmp directory if it doesn't exist
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # Paths for Docker volume
    tmp_input = tmp_dir / "input"
    tmp_output = tmp_dir / "output"
    
    try:
        # Copy input file to tmp/input
        print(f"Copying {input_file} to tmp/input...")
        shutil.copy2(input_file, tmp_input)
        
        # Remove old output if it exists
        if tmp_output.exists():
            if tmp_output.is_dir():
                shutil.rmtree(tmp_output)
            else:
                os.remove(tmp_output)
        
        # Run Docker command
        print("Running CICFlowMeter in Docker...")
        docker_cmd = [
            "docker", "run",
            "-v", f"{os.path.abspath(tmp_dir)}:/tmp/pcap",
            "mielverkerken/cicflowmeter",
            "/tmp/pcap/input",
            "/tmp/pcap/output"
        ]
        
        result = subprocess.run(docker_cmd)
        
        if result.returncode != 0:
            print(f"\nError: Docker command failed with exit code {result.returncode}")
            sys.exit(1)
        
        # Check if output was created
        if not tmp_output.exists():
            print(f"Error: Output file was not created in tmp/output")
            sys.exit(1)
        
        # Copy output to desired location
        print(f"Copying output to {output_file}...")
        if tmp_output.is_dir():
            # If output is a directory, find the CSV file inside
            csv_files = list(tmp_output.glob("*.csv"))
            if not csv_files:
                print("Error: No CSV file found in output directory")
                sys.exit(1)
            # Use the first CSV file found
            shutil.copy2(csv_files[0], output_file)
        else:
            # If output is a file, copy it directly
            shutil.copy2(tmp_output, output_file)
        
        print(f"Success! Output saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up tmp directory (optional - comment out if you want to keep files)
        # shutil.rmtree(tmp_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    main()

