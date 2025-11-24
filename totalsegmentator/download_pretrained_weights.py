from time import sleep

from libs import download_pretrained_weights
from config import setup_totalseg, set_config_key

if __name__ == "__main__":
    """
    Download all pretrained weights (without commercial models)
    """
    for task_id in [291, 292, 293, 294, 295, 297, 298, 258, 150, 260,
                    315, 299, 300, 850, 851, 852, 853, 775, 776, 777, 778,
                    779, 351, 913, 789, 527, 552, 570, 576, 115, 952, 113,
                    343]:

        setup_totalseg()
        set_config_key("statistics_disclaimer_shown", True)

        download_pretrained_weights(task_id)
        sleep(5)
