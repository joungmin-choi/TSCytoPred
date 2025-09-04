import subprocess
import os
import shutil

if os.path.exists("./results"):
    shutil.rmtree("./results")

def test_tsCytoPred_runs():
    # Run your script with the given arguments
    result = subprocess.run(
        ["python3", "tsCytoPred.py", "./example", "./results", "3", "15", "10"],
        capture_output=True,
        text=True
    )
    
    # Check that it exits cleanly
    assert result.returncode == 0, f"Process failed: {result.stderr}"

    # Optionally, check if output files were created
    assert os.path.exists("./results"), "Results directory was not created"

