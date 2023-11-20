from time import sleep

from libs import download_pretrained_weights

if __name__ == "__main__":
    """
    Download all pretrained weights
    """
    for task_id in [291, 292, 293, 294, 295, 297, 298, 258, 150, 260, 503,
                    315, 299, 300]:
        download_pretrained_weights(task_id)
        sleep(5)
