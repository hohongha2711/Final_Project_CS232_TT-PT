import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from sklearn.cluster import KMeans
import time
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgba2rgb


st.markdown(
""" 
# Image Compression
""")

def comp_Kmeans(img, k):
    start = time.time()
    rows, cols = img.shape[0], img.shape[1]
    img2 = img.reshape(rows*cols, 3)
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(img2)

    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    compressed_image = compressed_image.reshape(rows, cols, 3)
    end = time.time()
    io.imsave(r'result_kmeans.jpg',compressed_image)
    return compressed_image, round(end - start,5)


def svd(img,k):
    start = time.time()
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    Ur, Sr, Vr = np.linalg.svd(r, full_matrices = True)
    Ug, Sg, Vg = np.linalg.svd(g, full_matrices = True)
    Ub, Sb, Vb = np.linalg.svd(b, full_matrices = True)

    com_r = Ur[:, :k] @ np.diag(Sr[:k]) @ Vr[:k, :]
    com_g = Ug[:, :k] @ np.diag(Sg[:k]) @ Vg[:k, :]
    com_b = Ub[:, :k] @ np.diag(Sb[:k]) @ Vb[:k, :]

    compressed_image = np.zeros(img.shape[0:3])
    compressed_image[:,:,0] = com_r
    compressed_image[:,:,1] = com_g
    compressed_image[:,:,2] = com_b

    compressed_image_final = np.array(compressed_image/np.amax(compressed_image)*255, np.int32)
    compressed_image_final = compressed_image_final.astype('uint8')
    end = time.time()
    
    io.imsave(r'result_svd.jpg', compressed_image_final)
    return compressed_image_final, round(end - start,5)

def show_reusult(path1,path2,t):
    img = io.imread(path1)
    size_img = os.path.getsize(path1)

    img2 = io.imread(path2)
    size_img2 = os.path.getsize(path2)

    st.write("SSIM: ",round(ssim(img,img2,multichannel=True,channel_axis = -1),4))
    st.write("Ratio: ",round((size_img/size_img2),2))
    st.write("Time: ",t)

def svd_compare(img,path):

    comps = [1, 5, 20, 30,60,100, 200, 300]
    rows = [st.container() for _ in range(3)]
    cols_per_row = [r.columns(3) for r in rows]
    cols = [column for row in cols_per_row for column in row]
    for i in range(len(comps)+1):
        if(i == len(comps)):
            cols[i].image(img,caption='Actual image')
        else:
            start = time.time()
            r = img[:,:,0]
            g = img[:,:,1]
            b = img[:,:,2]

            Ur, Sr, Vr = np.linalg.svd(r, full_matrices = True)
            Ug, Sg, Vg = np.linalg.svd(g, full_matrices = True)
            Ub, Sb, Vb = np.linalg.svd(b, full_matrices = True)
            com_r = Ur[:, :comps[i]] @ np.diag(Sr[:comps[i]]) @ Vr[:comps[i], :]
            com_g = Ug[:, :comps[i]] @ np.diag(Sg[:comps[i]]) @ Vg[:comps[i], :]
            com_b = Ub[:, :comps[i]] @ np.diag(Sb[:comps[i]]) @ Vb[:comps[i], :]

            compressed_image = np.zeros(img.shape)
            compressed_image[:,:,0] = com_r
            compressed_image[:,:,1] = com_g
            compressed_image[:,:,2] = com_b

            compressed_image_final = np.array(compressed_image/np.amax(compressed_image)*255, np.int32)
            compressed_image_final = compressed_image_final.astype('uint8')
            end = time.time()
            path2 = 'compressed_svd_' + str(comps[i]) +'.jpg'
            io.imsave(path2, compressed_image_final)
            img1 = io.imread(path)
            size_img1 = os.path.getsize(path)

            img2 = io.imread(path2)
            size_img2 = os.path.getsize(path2)

            cols[i].image(img2, caption = f'K = {comps[i]}\n|\
                     Compression Ratio =  {round(size_img1/size_img2,2)}\n|\
                     SSIM: {round(ssim(img1,img2,multichannel=True,channel_axis = -1),4)}\n|\
                     Time: {round(end - start,5)}')


def kmeans_compare(img, path):
    img_actual = img
    rows, cols = img.shape[0], img.shape[1]
    img = img.reshape(rows*cols, 3)
    clusters = [2,4,8,16,32,64,100,128]
    rowss = [st.container() for _ in range(3)]
    cols_per_row = [r.columns(3) for r in rowss]
    colss = [column for row in cols_per_row for column in row]
    for i in range(len(clusters)+1):
        if(i == len(clusters)):
           colss[i].image(img_actual,caption='Actual image')
        else:
            start = time.time()

            kmeans = KMeans(n_clusters = clusters[i])
            kmeans.fit(img)

            compressed_image = kmeans.cluster_centers_[kmeans.labels_]
            compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

            # Reshape the image to the original dimension
            compressed_image_final = compressed_image.reshape(rows, cols, 3)
            end = time.time()
            path2 = 'compressed_kmeans_' + str(clusters[i]) +'.jpg'
            io.imsave(path2, compressed_image_final)
            img1 = io.imread(path)
            size_img1 = os.path.getsize(path)

            img2 = io.imread(path2)
            size_img2 = os.path.getsize(path2)

            colss[i].image(img2, caption = f'K = {clusters[i]}\n|\
                     Compression Ratio =  {round(size_img1/size_img2,2)}\n|\
                     SSIM: {round(ssim(img1,img2,multichannel=True,channel_axis = -1),4)}\n|\
                     Time: {round(end - start,5)}')

uploaded_file = st.file_uploader("Choose a file")
img = 0
img_path = ''
opt = ('SVD', 'Kmeans', 'Comparison SVD and Kmeans')
genre = st.radio("Select Algorithm", opt)
if uploaded_file is not None:
    # To read file as  and write to local disk:

    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    img_path = uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(bytes_data)
    img = io.imread(img_path)
    
    btn = st.button('Run')
    if btn:
        if genre == 'SVD':
            st.write('You selected SVD.')
            svd_compare(img, img_path)
        elif genre == 'Kmeans':
            st.write("You selected Kmeans.")
            kmeans_compare(img,img_path)
        else:
            st.write("You selected Comparison SVD and Kmeans.")
            result1,time1 = comp_Kmeans(img, 64)
            result2,time2 = svd(img, 200)
            col1,col2,col3 = st.columns(3)
            with col1:
                st.write("Ảnh gốc")
                st.image(img)
            with col2:
                st.write("After Compression with SVD")
                st.image(result2)
                show_reusult(img_path,r'result_svd.jpg',time2)
            with col3:
                st.write("After Compression with Kmeans")
                st.image(result1)
                show_reusult(img_path,r'result_kmeans.jpg',time1)
