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
    # カスタムフォントのパスを設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "fonts", "DejaVuSansMono.ttf") 

    # フォントの読み込み
    try:
        font_size = 60  
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        st.warning("指定されたフォントが見つからないため、デフォルトフォントを使用します。")
        font = ImageFont.load_default()
    
    # ロゴ画像のパスを設定
    logo_path = os.path.join(script_dir, "img", "logo.png")
    
    y_offset = 10  # 初期のy座標
    
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_size = (800, 800)
            logo.thumbnail(logo_size, Image.Resampling.LANCZOS)
            
            # ロゴの位置（左上）
            position = (10, y_offset)
            
            # 画像にロゴを貼り付け
            image.paste(logo, position, logo)
            
            y_offset += logo.size[1] + 10  # ロゴの高さとマージンを追加
        except Exception as e:
            st.warning(f"ロゴの埋め込みに失敗しました: {e}")
    
    # タイムスタンプの取得
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # テキストの内容
    text = f"{timestamp}\nCount: {detection_count}"
    
    # テキストの位置
    x, y = 10, y_offset  # 左上の余白
    
    # テキストの色とストロークの設定
    text_color = (0, 0, 0)  # 黒
    stroke_color = (255, 255, 255)  # 白
    stroke_width = 2  # ストロークの太さ

    # テキストを描画
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
    # セッションステートの初期化
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'app':
        run_application()

def show_home_page():
    
    # メイン画像の表示
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_image_path = os.path.join(script_dir, "img/logo.png")
    main_image_path = os.path.join(script_dir, "img/main.jpg")
    
    if os.path.exists(main_image_path):
        logo_image = Image.open(logo_image_path)
        main_image = Image.open(main_image_path)
        st.image(logo_image, use_container_width=True)
        st.image(main_image, use_container_width=True)
    else:
        st.warning("メイン画像（main.jpg）が見つかりません")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("使ってみる"):
        st.session_state.page = 'app'
        st.rerun()

def run_application():
    st.set_page_config(layout="centered", page_title=st.secrets["NAME"], page_icon="img/r.ico")
    st.title(st.secrets["TITLE"])

    # セッションステートの初期化
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
        st.error(f"モデルディレクトリが見つかりません: {models_dir}")
        sys.exit()
    
    # .ptファイルを取得
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not model_files:
        st.error("モデルが見つかりません")
        sys.exit()
    
    # サイドバーでモデルを選択
    selected_model = st.sidebar.selectbox(
        "使用するモデルを選択",
        options=model_files,
        index=0,
        help="使用するモデルを選択してください"
    )
    
    model_path = os.path.join(models_dir, selected_model)
    model = YOLO(model_path)

    # サイドバーの設定
    st.sidebar.header("設定")
    
    # 入力サイズの選択
    input_size = st.sidebar.selectbox(
        "入力サイズ",
        options=[640, 1024, 1280],
        key='input_size',
        help="入力サイズが高いほど細かく検出します"
    )
    
    # ラベル表示のオンオフ
    show_labels = st.sidebar.checkbox(
        "ラベル表示",
        key='show_labels',
        help="ラベル表示のオンオフです"
    )
    
    # conf下限値のスライダー
    conf_threshold = st.sidebar.slider(
        "conf下限値",
        min_value=0.05,
        max_value=0.70,
        key='conf_threshold',
        step=0.05,
        help="conf下限値の閾値を設定します"
    )
    
    # NMSのスライダー
    nms_threshold = st.sidebar.slider(
        "NMS",
        min_value=0.05,
        max_value=0.70,
        key='nms_threshold',
        step=0.05,
        help="NMSの閾値を設定します"
    )
    
    # レスポンシブレイアウトの設定
    col1, col2 = st.columns([1, 3])

    # タブの作成
    tabs = st.tabs(["アップロード", "None"])
    
    with tabs[0]:
        st.header("画像をアップロード")
        uploaded_file = st.file_uploader(
            "JPG/PNG画像をアップロード",
            type=["jpg", "jpeg", "png"],
            help=""
        )
        if uploaded_file:
            original = Image.open(uploaded_file)
            st.session_state.original_image = resize_and_limit(original)
            st.session_state.full_resolution_image = original
    

    with tabs[1]:
        st.header("None")
    
    if st.session_state.full_resolution_image:
        st.subheader("切り抜き範囲を選択")
        st.caption("シャーレの全体を囲うように切り抜き範囲を選択してください")
        st.caption("画像が見切れてしまう場合は画面を横回転にしてご利用下さい")
        
        image = st.session_state.full_resolution_image
        
        # Cropperの設定
        cropped_image = st_cropper(
            image,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=(1, 1)
        )
        
        if cropped_image:
            # クロップ後の画像が正方形であることを確認
            cropped_image = ensure_square(cropped_image)
            
            # クロップ画像のリサイズ
            final_image = cropped_image.resize((1800, 1800), Image.Resampling.LANCZOS)
            
            st.subheader("プレビュー")
            st.image(final_image, width=300)
            st.write(f"現在選択中のモデル: {selected_model}")
            if st.button("検出開始"):
                with st.spinner("検出中..."):
                    # YOLOに画像を渡して推論
                    results = model(
                        source=final_image,
                        imgsz=input_size,
                        line_width=1,
                        conf=conf_threshold,
                        iou=nms_threshold,
                        max_det=1000
                    )
                    
                    # 検出数の取得
                    num_detections = len(results[0].boxes)
                    st.success(f"検出数: {num_detections}")
                    
                    # 検出結果の描画
                    if show_labels:
                        annotated_image = results[0].plot()
                    else:
                        annotated_image = results[0].plot(labels=False)  
                    
                    # 色チャネルをBGRからRGBに変換
                    annotated_image = annotated_image[:, :, ::-1]
                    
                    annotated_pil = Image.fromarray(annotated_image)
                    
                    # タイムスタンプと検出数を追加
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
                    help="検出結果の画像をダウンロードします",
                )
            
    else:
        st.info("上部のタブか画像をアップロードするか、カメラで撮影してください")
    
    st.markdown(
        """
        <style>
        /* 全体の背景と文字色 */
        body {
            background: #f0f2f6;
            color: #333333;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        /* タイトルのスタイル */
        .css-1v3fvcr {
            font-size: 2.5rem;
            color: #1B4F72;
            text-align: center;
            margin-bottom: 30px;
            font-weight: bold;
        }

        /* ヘッダーのスタイル */
        .css-18e3th9 {
            color: #2C3E50;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 1.2rem;
        }

        /* ボタンのスタイル */
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

        /* ダウンロードボタンのスタイル */
        #download-button {
            background-color: #16A085;
        }
        #download-button:hover {
            background-color: #148F77;
        }

        /* ナビゲーションバーのスタイル */
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

        /* Cropperのボックス色 */
        .cropper-view-box,
        .cropper-face {
            border: 2px solid #16A085 !important;
        }

        /* フッターのスタイルを非表示 */
        footer {
            visibility: hidden;
        }

        /* レスポンシブデザインの調整 */
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
            .css-1e5imcs { /* タブのスタイルクラス */
                font-size: 14px;
            }
            .stButton button {
                padding: 10px 20px;
                font-size: 14px;
            }
        }

        /* 画像のフレームとシャドウ */
        img {
            border: 4px solid #1B4F72;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }

        /* カードスタイルの追加 */
        .card {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        /* タブスタイルの改善 */
        .css-1e5imcs { /* タブのスタイルクラス */
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