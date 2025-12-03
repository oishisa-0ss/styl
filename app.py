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
    """Load YOLO weights once and reuse across reruns to keep memory flat."""
    return YOLO(model_path)


def load_image_bytes(uploaded_file):
    """Read an UploadedFile into raw bytes (smaller than keeping a PIL object in session)."""
    return uploaded_file.getvalue()


def bytes_to_pil(image_bytes: bytes):
    """Create a PIL image from raw bytes with EXIF orientation fixed."""
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
    """
    Keep image square and make sure the side length does not exceed max_side.
    Avoids upscaling to reduce memory footprint during inference.
    """
    image = ensure_square(image)
    if image.width > max_side:
        image = image.resize((max_side, max_side), Image.Resampling.LANCZOS)
    return image

def add_timestamp_and_detection_count(image, detection_count, model_name, input_size, conf_threshold, nms_threshold):
    # Draw simple text overlay in the top-left (legacy style)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "fonts", "Mono.ttf") 
    
    try:
        # Legacy: fixed-ish size scaled to image, smaller than panel version
        font_size = max(28, int(min(image.size) * 0.03))
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
            logo_size = (200, 200)
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
        
        # st.image(logo_image, use_container_width=True)
        # st.image(main_image, use_container_width=True)
        # st.image(sub_image, use_container_width=True)
        
        st.image(logo_image, use_column_width=True)
        st.image(main_image, use_column_width=True)
        st.image(sub_image, use_column_width=True)
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
        st.image(logo_image, use_column_width=True)
        # st.image(logo_image, use_container_width=True)
    else:
        st.warning(st.secrets["PIC_ERR"])
        
    #st.title(st.secrets["TITLE"])

    ATTENTION = st.secrets["PG_ATTENTION"]
    if ATTENTION:
        st.write(f'{ATTENTION}',
                 unsafe_allow_html=True)
    
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
    if 'nms_threshold' not in st.session_state:
        st.session_state.nms_threshold = 0.45

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models") 

    if not os.path.exists(models_dir):
        st.error(st.secrets["MODEL_DIRERR"])
        sys.exit()
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

    if not model_files:
        st.error(st.secrets["MODEL_ERR"])
        sys.exit()

    st.sidebar.header("検出モデル選択")

    model_labels = [os.path.splitext(f)[0] for f in model_files]

    selected_label = st.sidebar.selectbox(
        "使用するモデルを選択",
        options=model_labels,
        index=1,
        help=st.secrets["MODEL_HELP"]
    )
    selected_model = selected_label + ".pt"
    selected_index = model_labels.index(selected_label)
    if 'previous_selected_label' not in st.session_state or st.session_state.previous_selected_label != selected_label:
        st.session_state.previous_selected_label = selected_label
        if selected_index == 0:
            st.session_state.conf_threshold = 0.20
            st.session_state.input_size = 768
        elif selected_index == 1:
            st.session_state.conf_threshold = 0.35
            st.session_state.input_size = 1024

    model_path = os.path.join(models_dir, selected_model)
    model = load_model(model_path)

    st.sidebar.header("設定")
    
    input_size = st.sidebar.selectbox(
        "## 入力サイズ",
        options=[640, 768, 1024, 1280],
        key='input_size',
        help=st.secrets["INPUT_HELP"]
    )
    
    show_labels = st.sidebar.checkbox(
        "## ラベル表示",
        key='show_labels',
        help=st.secrets["LABEL_HELP"]
    )
    
    conf_threshold = st.sidebar.slider(
        "## conf下限値",
        min_value=0.05,
        max_value=0.70,
        key='conf_threshold',
        step=0.05,
        help=st.secrets["CONF_HELP"]
    )
    
    nms_threshold = st.sidebar.slider(
        "## NMS",
        min_value=0.05,
        max_value=0.70,
        key='nms_threshold',
        step=0.05,
        help=st.secrets["NMS_HELP"]
    )
    
    col1, col2 = st.columns([1, 3])

    tabs = st.tabs(["画像をアップロード", "カメラを起動"])
    
    with tabs[0]:
        st.header("画像をアップロード")
        uploaded_file = st.file_uploader(
            "JPG/PNG画像をアップロード",
            type=["jpg", "jpeg", "png"],
            help=st.secrets["UPLOAD_HELP"]
        )
        if uploaded_file:
            st.session_state.detection_result_bytes = None
            st.session_state.full_image_bytes = load_image_bytes(uploaded_file)
    

    with tabs[1]:
        st.header("カメラで撮影")
        st.caption(st.secrets["CMR_ATTENTION"])
        camera_file = st.camera_input(
            "接写してください。", 
            help=st.secrets["CAMERA_HELP"]
            )

        if camera_file:
            st.session_state.detection_result_bytes = None
            st.session_state.full_image_bytes = load_image_bytes(camera_file)
    
    if st.session_state.full_image_bytes:
        st.subheader("切り抜き範囲を選択")
        st.caption(st.secrets["CLOP_CAP1"])
        st.caption(st.secrets["CLOP_CAP2"])
        
        image = bytes_to_pil(st.session_state.full_image_bytes)
        image = resize_and_limit(image)

        cropped_image = st_cropper(
            image,
            realtime_update=True,
            box_color="#1B4F72",
            aspect_ratio=(1, 1)
            #stroke_width=6
        )
        
        if cropped_image:
            max_side = min(input_size, 1280)
            final_image = clamp_square(cropped_image, max_side)
            
            st.subheader("プレビュー")
            st.image(final_image, width=300)
            st.write(f"##### 検出モデル: {selected_label}")
            st.write(f"##### 入力サイズ: ×{input_size}")
            st.write(f"##### conf下限値: {conf_threshold:.2f}")
            st.write(f"##### NMS: {nms_threshold:.2f}")
            st.write(st.secrets["MODEL_CONF_CAP1"])
            st.write(st.secrets["MODEL_CONF_CAP2"])

            if st.button("### 検出開始"):
                with st.spinner("検出中..."):
                    results = model(
                        source=final_image,
                        imgsz=input_size,
                        line_width=1,
                        conf=conf_threshold,
                        iou=nms_threshold,
                        max_det=1000
                    )
                    
                    num_detections = len(results[0].boxes)
                    st.success(f"検出数: {num_detections}")
                    
                    if show_labels:
                        annotated_image = results[0].plot()
                    else:
                        annotated_image = results[0].plot(labels=False)  
                    
                    annotated_image = annotated_image[:, :, ::-1]
                    annotated_pil = Image.fromarray(annotated_image)

                    annotated_pil = add_timestamp_and_detection_count(
                        annotated_pil, 
                        num_detections, 
                        selected_label, 
                        input_size, 
                        conf_threshold, 
                        nms_threshold
                    )
                    
                    DOWNLOAD_EXPORT_SIZE = 1800  # legacy download resolution
                    export_image = annotated_pil
                    if annotated_pil.width < DOWNLOAD_EXPORT_SIZE:
                        export_image = annotated_pil.resize(
                            (DOWNLOAD_EXPORT_SIZE, DOWNLOAD_EXPORT_SIZE),
                            Image.Resampling.LANCZOS,
                        )
                    
                    result_buf = io.BytesIO()
                    export_image.save(result_buf, format="JPEG", quality=95)
                    st.session_state.detection_result_bytes = result_buf.getvalue()
                    del results
                    gc.collect()
            
            if st.session_state.detection_result_bytes:
                # 再描画時も画像を登録し直してメディアID欠損を防ぐ
                st.image(st.session_state.detection_result_bytes, caption="検出結果", use_column_width=True)
                st.download_button(
                    label="結果をダウンロード",
                    data=st.session_state.detection_result_bytes,
                    file_name=f"rksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg",
                    key="download-detection",
                    help=st.secrets["DOWNLOAD_HELP"],
                )
    
    st.markdown(
        """
        <style>
       
        body {
            background: #e0f7e9; 
            color: #333333;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        .css-1v3fvcr {
            font-size: 2.5rem;
            color: #1B4F72;
            text-align: center;
            margin-bottom: 30px;
            font-weight: bold;
        }

        .css-18e3th9 {
            color: #2C3E50;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 1.2rem;
        }

        .stButton button {
            background-color: #1B4F72;
            color: #FFFFFF;
            border: none;
            padding: 12px 30px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 25px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        .stButton button:hover {
            background-color: #154360;
            transform: scale(1.05);
        }

        #download-button {
            background-color: #16A085;
        }
        #download-button:hover {
            background-color: #148F77;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #1B4F72;
            padding: 20px 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .navbar a {
            color: #FFFFFF;
            text-decoration: none;
            margin: 0 20px;
            font-size: 18px;
            transition: color 0.3s;
        }
        .navbar a:hover {
            color: #AED6F1;
        }

        .cropper-view-box,
        .cropper-face {
            border: 3px solid #16A085 !important;
        }

        footer {
            visibility: hidden;
        }

        @media only screen and (min-width: 768px) {
            .stApp {
                padding: 50px 150px;
            }
        }
        @media only screen and (max-width: 767px) {
            .stApp {
                padding: 20px 20px;
            }
            img {
                max-width: 100% !important;
                height: auto !important;
            }
            .css-1e5imcs {
                font-size: 14px;
            }
            .stButton button {
                padding: 10px 20px;
                font-size: 14px;
            }
        }

        /* img {
            border: 4px solid #1B4F72;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        } */

        .card {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .css-1e5imcs { 
            font-size: 18px;
            background-color: #1B4F72;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 12px;
            margin: 0 10px;
        }
        .css-1e5imcs:hover {
            background-color: #2471A3;
            color: #D6EAF8;
        }

        .stCropper {
            max-width: 100%;
            height: auto;
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
