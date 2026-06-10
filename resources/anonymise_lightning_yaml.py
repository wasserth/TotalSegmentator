import sys
from pathlib import Path


SENSITIVE_KEYS = {"subjects_test", "subjects_train", "subjects_val"}


def is_top_level_key(line):
    if not line or line[0].isspace() or line.lstrip().startswith("#"):
        return False
    return ":" in line


def remove_sensitive_keys(lines):
    cleaned_lines = []
    removed_keys = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if is_top_level_key(line):
            key = line.split(":", 1)[0].strip()
            if key in SENSITIVE_KEYS:
                removed_keys.append(key)
                i += 1

                while i < len(lines):
                    next_line = lines[i]
                    if is_top_level_key(next_line):
                        break
                    i += 1
                continue

        cleaned_lines.append(line)
        i += 1

    return cleaned_lines, removed_keys


if __name__ == "__main__":
    """
    Remove subject IDs from Lightning hparams.yaml files.

    usage:
    cd /mnt/nvme/data/multiseg/weights_upload/lightning_models
    python ~/dev/TotalSegmentator/resources/anonymise_lightning_yaml.py ct_age_splitOrig_2d_ns5_effnetv2

    Remember to manually remove the events files and mean directory:
    find . -type f -name "events.out*" -delete
    find . -type d -name "mean" -prune -exec rm -rf -- {} +

    Then zip all:
    
    zip -r mr_weight_splitOrig_2d_ns5_effnetv2_fl1_se1.zip mr_weight_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r mr_size_splitOrig_2d_ns5_effnetv2_fl1_se1.zip mr_size_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r mr_age_splitOrig_2d_ns5_effnetv2_fl1_se1.zip mr_age_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r mr_sex_splitOrig_2d_ns5_effnetv2_fl1_se1.zip mr_sex_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r ct_weight_splitOrig_2d_ns5_effnetv2_fl1_se1.zip ct_weight_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r ct_size_splitOrig_2d_ns5_effnetv2_fl1_se1.zip ct_size_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r ct_age_splitOrig_2d_ns5_effnetv2_fl1_se1.zip ct_age_splitOrig_2d_ns5_effnetv2_fl1_se1;\
    zip -r ct_sex_splitOrig_2d_ns5_effnetv2_fl1_se1.zip ct_sex_splitOrig_2d_ns5_effnetv2_fl1_se1
    
    """
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    hparams_files = sorted(model_dir.rglob("hparams.yaml"))
    print(f"Nr of hparams.yaml files found: {len(hparams_files)}")

    for hparams_file in hparams_files:
        lines = hparams_file.read_text().splitlines(keepends=True)
        cleaned_lines, removed_keys = remove_sensitive_keys(lines)

        if removed_keys:
            hparams_file.write_text("".join(cleaned_lines))
            print(f"Anonymised: {hparams_file} (removed: {', '.join(removed_keys)})")
