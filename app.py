import streamlit as st
from face_blur import blur_faces_image, blur_faces_video
import os

st.set_page_config(layout="wide")
st.title('Face Blurring Tool')
st.write('Upload an image or video and select the blur type to blur faces.')

def clear_old_results():
    st.session_state.result_path = None

if 'result_path' not in st.session_state:
    st.session_state.result_path = None
if 'blur_type' not in st.session_state:
    st.session_state.blur_type = 'pixelate'

with st.sidebar:
    uploaded_file = st.file_uploader(
        "## Choose Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4'], 
        on_change=clear_old_results
    )

    st.selectbox(
        "Select Blur Type", 
        ['pixelate', 'blur', 'blackbox'],
        key='blur_type',
        help="Choose the visual effect to apply to detected faces."
    )

if uploaded_file is not None:
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_path = os.path.join(uploads_dir, 'input_file.' + file_ext)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    is_image = file_ext in ['jpg', 'jpeg', 'png']
    is_video = file_ext == 'mp4'

    with col1:
        st.subheader("Original")
        if is_image:
            st.image(file_path, use_container_width=True)
        elif is_video:
            st.video(file_path)

    if st.sidebar.button(f'Blur Faces in {("Image" if is_image else "Video")}', use_container_width=True):
        with st.spinner('Processing... Please wait.'):
            blur_type = st.session_state.blur_type
            if is_image:
                st.session_state.result_path = blur_faces_image(file_path, blur_type)
            elif is_video:
                st.session_state.result_path = blur_faces_video(file_path, blur_type)
            
            if not st.session_state.result_path:
                st.error("Processing failed. Please check the logs.")

    if st.session_state.result_path and os.path.exists(st.session_state.result_path):
        with col2:
            st.subheader("Blurred Result")
            if is_image:
                st.image(st.session_state.result_path, use_container_width=True)
                with open(st.session_state.result_path, "rb") as file:
                    st.download_button(
                        label="Download Blurred Image",
                        data=file,
                        file_name=f"blurred_{uploaded_file.name}",
                        mime=f"image/{file_ext}"
                    )
            elif is_video:
                st.video(st.session_state.result_path)
                with open(st.session_state.result_path, "rb") as file:
                    st.download_button(
                        label="Download Blurred Video",
                        data=file,
                        file_name=f"blurred_{uploaded_file.name}",
                        mime="video/mp4"
                    )
else:
    st.info("Please upload a file to begin.")
