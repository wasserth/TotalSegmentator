from time import sleep

from libs import download_pretrained_weights

if __name__ == "__main__":
    """
    Download all pretrained weights
    """
    for task_id in [251, 252, 253, 254, 255, 256]:
        download_pretrained_weights(task_id)
        sleep(5)
