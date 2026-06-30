from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import imgkit
from PIL import Image


def to_date(date_as_int):
    if date_as_int:
        return datetime.strptime(str(date_as_int), "%Y%m%d").strftime("%d.%m.%Y")
    else:
        return ""


def now():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def rm_underscores(s):
    """
    remove underscores from string
    """
    return s.replace("_", " ")


def generate_html(templates_path, template_file, template_vars, output_file, width):

    # templates_path = str(Path(__file__).absolute().parent)
    # print(f"templates_path: {templates_path}")

    env = Environment(loader=FileSystemLoader(templates_path))
    env.filters["to_date"] = to_date
    env.globals["now"] = now
    env.globals["rm_underscores"] = rm_underscores
    template = env.get_template(template_file)

    template_vars["styles_path"] = f"{templates_path}/styles.css"
    template_vars["usb_logo_path"] = f"{templates_path}/usb_logo_white.svg"
    template_vars["empty_img_path"] = f"{templates_path}/empty_bg.png"
    template_vars["logo_path"] = f"{templates_path}/logo_black.svg"
    html_out = template.render(template_vars)

    # save html to file - for debugging
    with open(str(output_file) + ".html", "w") as f:
        f.write(html_out)

    # increase dpi: not working properly
    # width_px = 1400
    # zoom = 2  
    # width = width_px * zoom

    zoom = 1
    # by setting a option to empty string it gets activated
    imgkit.from_string(html_out, output_file,
                       options={"xvfb": "", "format": "png", "width": str(width),
                                "enable-local-file-access": "", "zoom": str(zoom), 
                                "quality": "75", "quiet": ""})

    return Image.open(output_file)
