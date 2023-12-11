import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.naive_bayes import MultinomialNB
import pickle

selected=option_menu(
    menu_title=None,
    options=['Data', 'Implementasi','Me'],
    default_index=0,
    orientation='horizontal',
    menu_icon=None,
    styles={
    "nav-link":{
        "font-size":"12px",
        "text-align":"center",
        "margin":"5px",
        "--hover-color":"pink",},
    "nav-link-selected":{
        "background-color":"purple"},
    })
if selected=='Data':
    st.title('APLIKASI KLASIFIKASI CAR EVALUATION')
    image_path = "carevaluation.jpg"
    
    # Menampilkan gambar dari file lokal
    st.image(image_path, caption="gambar MOBIL", use_column_width=True, output_format="auto")
    ```python
import streamlit as st

# Judul
st.title("Analisis Dataset Car Evaluation")

st.write("Deskripsi Data")
st.write("Data yang digunakan diambil dari website UCI REPOSITORY: [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)<br>")
st.write("Dataset memiliki 7 kolom dan terdiri dari 1728 baris data. Deskripsi dataset menyatakan bahwa dataset car evaluation bersifat kategorikal dan tidak memiliki missing value.<br>")

# Jumlah Sampel
st.header("1. Jumlah Sampel:")
st.write("Dataset \"car evaluation\" terdiri dari 7 kolom (6 fitur ciri dan Label) serta dataset car evaluation memiliki 1728 baris data secara keseluruhan.<br>")
st.write("Jumlah sampel berdasarkan kelas:<br>")
st.write("- unacc = 1210<br>")
st.write("- acc = 384<br>")
st.write("- good = 69<br>")
st.write("- vgood = 65<br>")

# One-Hot Encoding
st.header("2. One-Hot Encoding:")
st.write("Semua fitur pada dataset car evaluation bertype data kategorikal. Oleh karena itu, akan dilakukan one-hot encoding pada proses preprocessing.<br>")
st.write("One-hot encoding merupakan metode representasi data kategorikal dalam bentuk yang dapat diolah oleh model pembelajaran mesin.<br>")
```python
# Penjelasan Fitur
st.header("Penjelasan Fitur:")
st.write("1. **buying (Pembelian):** Tingkat kelas mobil berdasarkan harga pembelian, dengan nilai yang mungkin termasuk \"vhigh\" (sangat tinggi), \"high\" (tinggi), \"med\" (sedang), dan \"low\" (rendah).<br>")
st.write("2. **maint (Pemeliharaan):** Tingkat kelas mobil berdasarkan biaya pemeliharaan, dengan nilai yang mungkin sama seperti pada atribut pembelian.<br>")
st.write("3. **doors (Jumlah Pintu):** Jumlah pintu pada mobil, dengan nilai yang mungkin termasuk 2, 3, 4, atau \"5more\" (lebih dari 5 pintu).<br>")
st.write("4. **persons (Kapasitas Penumpang):** Kapasitas penumpang pada mobil, dengan nilai yang mungkin termasuk 2, 4, atau \"more\" (lebih dari 4 penumpang).<br>")
st.write("5. **lug_boot (Kapasitas Bagasi):** Ukuran bagasi pada mobil, dengan nilai yang mungkin termasuk \"small\" (kecil), \"med\" (sedang), atau \"big\" (besar).<br>")
st.write("6. **safety (Keamanan):** Tingkat keamanan mobil, dengan nilai yang mungkin termasuk \"low\" (rendah), \"med\" (sedang), atau \"high\" (tinggi).<br>")
st.write("7. **Class Labels (Label Kelas):**<br>")
st.write("- unacc: Tidak dapat diterima,<br>")
st.write("- acc: Dapat diterima,<br>")
st.write("- good: Baik,<br>")
st.write("- vgood: Sangat baik.<br>")
if selected=='Implementasi':
    kolom=['buying_high','buying_low','buying_med','buying_vhigh','maint_high','maint_low','maint_med','maint_vhigh',
'doors_2','doors_3','doors_4','doors_5more','persons_2','persons_4','persons_more','lug_boot_big','lug_boot_med','lug_boot_small','safety_high','safety_low','safety_med']
    st.title('CAR EVALUATION')

    df = pd.DataFrame(data=[[0]*len(kolom)], columns=kolom)

    buying=['Silahkan Pilih','Very High','High','Med','Low']
    maint=['Silahkan Pilih','Very High','High','Med','Low']
    doors=['Silahkan Pilih','2','3','4','5 More']
    persons=['Silahkan Pilih','2','4','More']
    lug_boot=['Silahkan Pilih','Small','Med','Big']
    safety=['Silahkan Pilih','low','Med','High']

    buy=st.selectbox('PILIH Kelas Mobil',buying)
    if buy=='Very High':
        df['buying_vhigh']=1
    if buy=='High':
        df['buying_high']=1
    if buy=='Med':
        df['buying_med']=1
    if buy=='Low':
        df['buying_low']=1


    maint1=st.selectbox('PILIH Tingkat Pemeliharaan',maint)
    if maint1=='Very High':
        df['maint_vhigh']=1
    if maint1=='High':
        df['maint_high']=1
    if maint1=='Med':
        df['maint_med']=1
    if maint1=='Low':
        df['maint_low']=1

    dr=st.selectbox('PILIH Jumlah Pintu',doors)
    if dr=='2':
        df['doors_2']=1
    if dr=='3':
        df['doors_3']=1
    if dr=='4':
        df['doors_4']=1
    if dr=='5 More':
        df['doors_5more']=1

    pers=st.selectbox('PILIH Kapasitas Persons',persons)
    if pers =='2':
        df['persons_2']=1
    if pers =='4':
        df['persons_4']=1
    if pers =='More':
        df['persons_more']=1

    lb=st.selectbox('PILIH Lug Boot',lug_boot)
    if lb =='Big':
        df['lug_boot_big']=1
    if lb =='Med':
        df['lug_boot_med']=1
        df['lug_boot_small']=1

    sf=st.selectbox('PILIH Jenis Safety',safety)
    if sf =='High':
        df['safety_high']=1
    if sf =='Low':
        df['safety_low']=1
    if sf =='Med':
        df['safety_med']=1

            
    button=st.button('KLASIFIKASI',use_container_width=500,type='primary')
    if button:
        if buying!='Silahkan Pilih'and maint!='Silahkan Pilih'and doors!='Silahkan Pilih'and persons!='Silahkan Pilih'and lug_boot!='Silahkan Pilih'and safety!='Silahkan Pilih':
            st.write(df)
            with open('car_naive.pkl', 'rb') as naive:
                naiveby= pickle.load(naive)
            predik=naiveby.predict(df)
            for predi in predik:
                st.write('Kelas : ', predi)
        else:
            st.write('Mohon Isi semua Kolom Pertanyaan')
if selected=='Me':
    st.write("Kelompok 9")
    st.write("Machine Learning")
    st.write("SEMESTER GANJIL 2023")
