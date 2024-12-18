import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from streamlit_cropper import st_cropper
from datetime import datetime
from ultralytics import YOLO
import os
import sys

def resize_and_limit(image, max_size=1200):
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def ensure_square(image):
    if image.width != image.height:
        min_side = min(image.width, image.height)
        return image.crop((0, 0, min_side, min_side))
    return image

def add_timestamp_and_detection_count(image, detection_count):
    draw = ImageDraw.Draw(image)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "fonts", "DejaVuSansMono.ttf") 
    
    try:
        font_size = 60  
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        st.warning(st.secrets["FONT_WARNING"])
        font = ImageFont.load_default()
    
    logo_path = os.path.join(script_dir, "img", "logo.png")
    
    y_offset = 10  
    
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_size = (800, 800)
            logo.thumbnail(logo_size, Image.Resampling.LANCZOS)
            
            position = (10, y_offset)
            
            image.paste(logo, position, logo)
            
            y_offset += logo.size[1] + 10
        except Exception as e:
            st.warning(st.secrets["LOGO_WANING"])
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text = f"{timestamp}\nCount: {detection_count}"
    
    x, y = 10, y_offset  
    
    text_color = (0, 0, 0)  
    stroke_color = (255, 255, 255)  
    stroke_width = 2  

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )
    
    return image

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
    
    if os.path.exists(main_image_path):
        logo_image = Image.open(logo_image_path)
        main_image = Image.open(main_image_path)
        st.image(logo_image, use_container_width=True)
        st.image(main_image, use_container_width=True)
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
        
    st.title(st.secrets["TITLE"])

    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'detection_result' not in st.session_state:
        st.session_state.detection_result = None
    if 'input_size' not in st.session_state:
        st.session_state.input_size = 1024
    if 'show_labels' not in st.session_state:
        st.session_state.show_labels = False
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.20
    if 'nms_threshold' not in st.session_state:
        st.session_state.nms_threshold = 0.45
    if 'full_resolution_image' not in st.session_state:
        st.session_state.full_resolution_image = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models") 
    if not os.path.exists(models_dir):
        st.error(st.secrets["MODEL_DIRERR"])
        sys.exit()
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not model_files:
        st.error(st.secrets["MODEL_ERR"])
        sys.exit()
    
    selected_model = st.sidebar.selectbox(
        "使用するモデルを選択",
        options=model_files,
        index=0,
        help=st.secrets["MODEL_HELP"]
    )
    
    model_path = os.path.join(models_dir, selected_model)
    model = YOLO(model_path)

    st.sidebar.header("設定")
    
    input_size = st.sidebar.selectbox(
        "入力サイズ",
        options=[640, 1024, 1280],
        key='input_size',
        help=st.secrets["INPUT_HELP"]
    )
    
    show_labels = st.sidebar.checkbox(
        "ラベル表示",
        key='show_labels',
        help=st.secrets["LABEL_HELP"]
    )
    
    conf_threshold = st.sidebar.slider(
        "conf下限値",
        min_value=0.05,
        max_value=0.70,
        key='conf_threshold',
        step=0.05,
        help=st.secrets["CONF_HELP"]
    )
    
    nms_threshold = st.sidebar.slider(
        "NMS",
        min_value=0.05,
        max_value=0.70,
        key='nms_threshold',
        step=0.05,
        help=st.secrets["NMS_HELP"]
    )
    
    col1, col2 = st.columns([1, 3])

    tabs = st.tabs(["アップロード", "None"])
    
    with tabs[0]:
        st.header("画像をアップロード")
        uploaded_file = st.file_uploader(
            "JPG/PNG画像をアップロード",
            type=["jpg", "jpeg", "png"],
            help=st.secrets["UPLOAD_HELP"]
        )
        if uploaded_file:
            original = Image.open(uploaded_file)
            st.session_state.original_image = resize_and_limit(original)
            st.session_state.full_resolution_image = original
    

    with tabs[1]:
        st.header("None")
    
    if st.session_state.full_resolution_image:
        st.subheader("切り抜き範囲を選択")
        st.caption(st.secrets["CLOP_CAP1"])
        st.caption(st.secrets["CLOP_CAP2"])
        
        image = st.session_state.full_resolution_image
        
        cropped_image = st_cropper(
            image,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=(1, 1)
        )
        
        if cropped_image:
            cropped_image = ensure_square(cropped_image)
            final_image = cropped_image.resize((1800, 1800), Image.Resampling.LANCZOS)
            
            st.subheader("プレビュー")
            st.image(final_image, width=300)
            st.write(f"現在選択中のモデル: {selected_model}")
            st.write(st.secrets["MODEL_CONF_CAP1"])
            st.write(st.secrets["MODEL_CONF_CAP2"])
            if st.button("検出開始"):
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
                    
                    annotated_pil = add_timestamp_and_detection_count(annotated_pil, num_detections)
                    
                    st.session_state.detection_result = annotated_pil
                    st.image(annotated_pil, caption="検出結果", width=500)
            
            if st.session_state.detection_result:
                buf = io.BytesIO()
                st.session_state.detection_result.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="結果をダウンロード",
                    data=buf.getvalue(),
                    file_name=f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg",
                    key="download-detection",
                    help=st.secrets["DOWNLOAD_HELP"],
                )
            
    else:
        st.info("上部のタブか画像をアップロードするか、カメラで撮影してください")
    
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

        img {
            border: 4px solid #1B4F72;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }

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
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 