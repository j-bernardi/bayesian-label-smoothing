import os
import sys
import shutil
import pprint
import traceback

from main import main


REPEATS = [i for i in range(5)]  # don't overwrite if wish to add more
SUBDIR = os.path.join("experiments", "smoothing_optim")

# Function name : parameter
smooths = {
    "fixed_uniform_smoothing": [
        "0.800", "0.825", "0.850", "0.875", "0.900", "0.925", "0.950", "0.975"
    ],
    "fixed_adjacent_smoothing": [
        "0.800", "0.825", "0.850", "0.875", "0.900", "0.925", "0.950", "0.975"
    ],
    "weighted_uniform_smoothing": [
        "0.800", "0.825", "0.850", "0.875", "0.900", "0.925", "0.950", "0.975"
    ],
    "weighted_adjacent_smoothing": [
        "0.800", "0.825", "0.850", "0.875", "0.900", "0.925", "0.950", "0.975"
    ],
}


if __name__ == "__main__":

    failed = {}

    # Sweep so that can start seeing results after 1 pass of r
    for r in REPEATS:
        for smoothing_type in smooths:
            for param in smooths[smoothing_type]:

                exp_dir = os.path.join(SUBDIR, f"{smoothing_type}_{param}_{r}")
                # No dots in dir name (for importlib, config loading)
                exp_dir = exp_dir.replace("0.", "")
                os.makedirs(exp_dir, exist_ok=True)
                print("\n\n***", exp_dir, "***\n\n")

                exp_config_file = shutil.copy(
                    os.path.join("experiments", "smoothing_optim_template.py"),
                    os.path.join(exp_dir, "config.py")
                )

                full_string = open(exp_config_file, "r").read()
                new_string = full_string.replace(
                    "FUNCTION_PLACEHOLDER", smoothing_type
                )
                new_string = new_string.replace("PARAMETER_PLACEHOLDER", param)
                # Overwrite config with new string
                with open(exp_config_file, "w") as f:
                    f.write(new_string)

                try:
                    success = main([exp_dir, "-d", "3"])
                    if not success:
                        failed[exp_dir] = "Returned prematurely"
                except Exception as e:
                    traceback.print_exc()
                    failed[exp_dir] = str(e)

    print(f"FAILURES:\n{pprint.pformat(failed)}")
