from time import sleep

from libs import download_pretrained_weights

if __name__ == "__main__":
    """
    Download all pretrained weights (without commercial models)
    """
    for task_id in [291, 292, 293, 294, 295, 297, 298, 258, 150, 260,
                    315, 299, 300, 850, 851, 852, 853, 775, 776, 777, 778,
                    779, 351, 913, 789, 527, 552]:
        download_pretrained_weights(task_id)
        sleep(5)
