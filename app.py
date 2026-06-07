import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import gc
from streamlit_cropper import st_cropper
from datetime import datetime
from ultralytics import YOLO
import os
import sys
import pytz


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)


def load_image_bytes(uploaded_file):
    return uploaded_file.getvalue()


def bytes_to_pil(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image)


def resize_and_limit(image, max_size=1200):
    image = ImageOps.exif_transpose(image)
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        return resized_image
    return image


def ensure_square(image):
    if image.width != image.height:
        min_side = min(image.width, image.height)
        return image.crop((0, 0, min_side, min_side))
    return image


def clamp_square(image, max_side: int):
    image = ensure_square(image)
    if image.width > max_side:
        image = image.resize((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def add_timestamp_and_detection_count(image, detection_count, model_name, input_size, conf_threshold, nms_threshold):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "fonts", "Mono.ttf")
    
    try:
        font_size = max(18, int(min(image.size) * 0.022))
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        st.warning(st.secrets["FONT_WARNING"])
        font = ImageFont.load_default()
    
    logo_path = os.path.join(script_dir, "img", "logo.png")
    y_offset = 10
    
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    timestamp = datetime.now(tokyo_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{timestamp}")

    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_width = int(min(image.size) * 0.25)
            logo_size = (logo_width, logo_width)
            logo.thumbnail(logo_size, Image.Resampling.LANCZOS)
            position = (10, y_offset)
            image.paste(logo, position, logo)
            y_offset += logo.size[1] + 10
        except Exception as e:
            st.warning(st.secrets["LOGO_WANING"])

    text = (
        f"{timestamp}\n"
        f"Count: {detection_count}\n"
        f"Model: {model_name}\n"
        f"Input: ×{input_size}\n"
        f"Conf : {conf_threshold:.2f}\n"
        f"NMS  : {nms_threshold:.2f}"
    )
    
    x, y = 10, y_offset
    text_color = (0, 0, 0, 255)
    stroke_color = (255, 255, 255, 255)
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        spacing=int(font_size * 0.25),
        stroke_width=1,
        stroke_fill=stroke_color,
        align="left",
    )

    footer_text = "These detection results are for reference only."
    
    try:
        footer_font_size = max(12, int(min(image.size) * 0.012))
        footer_font = ImageFont.truetype(font_path, size=footer_font_size)
    except IOError:
        footer_font = ImageFont.load_default()

    footer_color = (128, 128, 128, 255)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), footer_text, font=footer_font, spacing=4)
    text_height = bottom - top
    
    footer_x = 10
    footer_y = image.height - text_height - 20

    draw.multiline_text(
        (footer_x, footer_y),
        footer_text,
        font=footer_font,
        fill=footer_color,
        spacing=4,
        align="left",
    )

    return image.convert("RGB")


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'app':
        run_application()


def show_home_page():
    st.set_page_config(
        layout="centered",
        page_title=st.secrets["NAME"],
        page_icon="img/r.ico"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_image_path = os.path.join(script_dir, "img/logo.png")
    main_image_path = os.path.join(script_dir, "img/main.jpg")
    sub_image_path = os.path.join(script_dir, "img/sub.jpg")

    if os.path.exists(main_image_path):
        logo_image = Image.open(logo_image_path)
        main_image = Image.open(main_image_path)
        sub_image = Image.open(sub_image_path)
        
        st.image(logo_image, use_container_width=True)
        st.image(main_image, use_container_width=True)
        st.image(sub_image, use_container_width=True)
    else:
        st.warning(st.secrets["PIC_ERR"])
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button(st.secrets["USE_BUTTON"]):
        st.session_state.page = 'app'
        st.rerun()


def run_application():
    st.set_page_config(
        layout="centered",
        page_title=st.secrets["NAME"],
        page_icon="img/r.ico"
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_image_path = os.path.join(script_dir, "img/logo.png")

    if os.path.exists(logo_image_path):
        logo_image = Image.open(logo_image_path)
        st.image(logo_image, use_container_width=True)
    else:
        st.warning(st.secrets["PIC_ERR"])
        
    ATTENTION = st.secrets["PG_ATTENTION"]
    if ATTENTION:
        st.write(f'{ATTENTION}', unsafe_allow_html=True)
    
    if 'full_image_bytes' not in st.session_state:
        st.session_state.full_image_bytes = None
    if 'detection_result_bytes' not in st.session_state:
        st.session_state.detection_result_bytes = None
    if 'input_size' not in st.session_state:
        st.session_state.input_size = 1024
    if 'show_labels' not in st.session_state:
        st.session_state.show_labels = False
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.20
    if 'nms_threshold' not in
