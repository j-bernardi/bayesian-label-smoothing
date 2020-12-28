import os
import sys
import shutil
import pprint

from main import main


REPEATS = [i for i in range(5)]  # don't overwrite if wish to add more
SUBDIR = os.path.join("experiments", "smoothing_optim")

# Function name : parameter
smooths = {
    "fixed_uniform_smoothing":
        ["0.80", "0.825", "0.85", "0.875", "0.90", "0.925", "0.95", "0.975"],
    "fixed_adjacent_smoothing":
        ["0.80", "0.825", "0.85", "0.875", "0.90", "0.925", "0.95", "0.975"],
    "weighted_uniform_smoothing":
        ["0.80", "0.825", "0.85", "0.875", "0.90", "0.925", "0.95", "0.975"],
    "weighted_adjacent_smoothing":
        ["0.80", "0.825", "0.85", "0.875", "0.90", "0.925", "0.95", "0.975"],
}


if __name__ == "__main__":

    failed = {}

    # Sweep so that can start seeing results after 1 pass of r
    for r in REPEATS:
        for smoothing_type in smooths:
            for param in smooths[smoothing_type]:

                exp_dir = os.path.join(SUBDIR, f"{smoothing_type}_{param}_{r}")
                os.makedirs(exp_dir, exist_ok=True)
                print("\n\n***", exp_dir, "***\n\n")

                config = shutil.copy(
                    os.path.join("experiments", "smoothing_optim_template.py"),
                    os.path.join(exp_dir, "config.py")
                )

                full_string = open(config, "r").read()
                new_string = full_string.replace(
                    "FUNCTION_PLACEHOLDER", smoothing_type
                )
                new_string = new_string.replace("PARAMETER_PLACEHOLDER", param)
                # Overwrite config with new string
                with open(config, "w") as f:
                    f.write(new_string)

                try:
                    success = main([exp_dir, "-d", "3"])
                    if not success:
                        failed[exp_dir] = "Returned prematurely"
                except Exception as e:
                    failed[exp_dir] = str(e)

    print(f"FAILURES:\n{pprint.pformat(failed)}")
