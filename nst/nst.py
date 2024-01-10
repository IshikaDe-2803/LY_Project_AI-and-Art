import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from gan import G_BA
from cartoonizer import G_model
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from nst_script import main_style_transfer
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

Tensor = torch.Tensor

def load_checkpoint(ckpt_path, map_location=torch.device('cpu')):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

st.title('NeuralArt')
st.write("Upload an image and get generated stylized images!")
st.sidebar.write("## Upload and download :gear:")

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def stylize(image_url, style, epochs):
    if style_option == 'Custom Style':
        opt =  Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        result = main_style_transfer(image_url, style, opt, epochs)
        result.save(os.path.join('./', 'stylized.png'))
        return
    if style_option == 'Ukiyoe':
        g = load_checkpoint('./ukiyoe.ckpt')
    if style_option == 'Van Gogh':
        g = load_checkpoint('./vangogh.ckpt')
    if style_option == 'Monet':
        g = load_checkpoint('./monet.ckpt')
    if style_option == 'CycleGAN Indian':
        g = load_checkpoint('./current_80.ckpt')
    if style_option == 'Indian Comic Style':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load('./comic.pth', map_location=torch.device(device))
        G_inference = G_model
        G_inference.load_state_dict(checkpoint['g_state_dict'])
        image_path = image_url
        your_image_size = 256  # Replace with the size expected by your generator model
        preprocess = transforms.Compose([
            transforms.Resize(your_image_size),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        result_image_best_checkpoint = G_inference(image_tensor)
        result_image_pil = transforms.ToPILImage()(result_image_best_checkpoint.squeeze(0).cpu())
        result_image_pil.save(os.path.join('./', 'stylized.png'))
        return

    G_BA.load_state_dict(g['G_BA'])
    generate_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_image = transforms.ToPILImage()

    G_BA.eval()

    imgs = []
    img = Image.open(image_url)
    img = generate_transforms(img)
    imgs.append(img)
    imgs = torch.stack(imgs, 0).type(Tensor)
    fake_imgs = G_BA(imgs).detach().cpu()
    for j in range(fake_imgs.size(0)):
        img = fake_imgs[j].squeeze().permute(1, 2, 0)
        img_arr = img.numpy()
        img_arr = (img_arr - np.min(img_arr)) * 255 / (np.max(img_arr) - np.min(img_arr))
        img_arr = img_arr.astype(np.uint8)        
        img = Image.fromarray(img_arr)
        img.save(os.path.join('./', 'stylized.png'))

def fix_image(content, style, isContent = False, epochs = 10):
    content_image = Image.open(content)
    col1.write("Input Image :camera:")
    col1.image(content_image)
    if isContent:
        stylize(content, style, epochs)
        output_image = Image.open('./stylized.png')
        col2.write("Stylized Image :art:")
        col2.image(output_image)

        st.sidebar.markdown("\n")

        output_image = convert_image(output_image)
        st.sidebar.download_button( label="Download image",
                                    data=output_image,
                                    file_name="stylized.png",
                                    mime="image/png")
     


col1, col2 = st.columns(2)

style_option = st.sidebar.selectbox(
    'Select the style that you want:',
    ('Van Gogh', 'Ukiyoe', 'Monet', 'Indian Comic Style', 'Custom Style','CycleGAN Indian'))

content_image = st.sidebar.file_uploader("Upload a input image :camera:", type=["png", "jpg", "jpeg"], key=1)

style_image = None

if style_option == 'Custom Style':
    print(style_option)
    epochs = st.sidebar.number_input("Enter number of training steps (Note: Every 5 epochs takes almost a minute to run. This may vary depending on image quality.):", value=0, step=1)
    style_image = st.sidebar.file_uploader("Upload a style image :camera:", type=["png", "jpg", "jpeg"], key=2  )
     
if content_image is not None and style_image is not None and style_option == 'Custom Style':
    if epochs == 0:
        epochs = 10
    fix_image(content=content_image, style=style_image, isContent=True, epochs=epochs)
elif content_image is not None and style_option != 'Custom Style':
    fix_image(content=content_image, style=style_image, isContent=True)
else:
    file_ = open("./nstvid.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
        unsafe_allow_html=True,
    )





