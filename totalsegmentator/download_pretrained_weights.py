from time import sleep

from totalsegmentator.libs import download_pretrained_weights, get_weights_dir

if __name__ == "__main__":
    """
    Download all pretrained weights
    """
    config_dir = get_weights_dir()
    for task_id in [291, 292, 293, 294, 295, 297, 298, 258, 150, 260, 503,
                    315, 299, 300, 730, 731, 732, 733, 775, 776, 777, 778,
                    779]:
        download_pretrained_weights(task_id, config_dir, source="github")
        sleep(5)
