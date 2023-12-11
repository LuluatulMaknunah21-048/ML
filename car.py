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
    st.write("Data yang digunakan diambil dari website UCI REPOSITORY : https://archive.ics.uci.edu/dataset/73/mushroom. Dataset mushroom digunakan untuk menganalisis faktor-faktor yang membedakan jamur yang dapat dikonsumsi dengan yang beracun berdasarkan berbagai ciri jamur atau karakteristik jamur.")

    st.write("**Deskripsi Dataset:**")
    st.write("1. **Jumlah Sampel:**")
    st.write("   Dataset 'Mushroom' terdiri dari 8124 baris data secara keseluruhan, atau jika berdasarkan kelas 4208 data berkelas e dan 3916 data berkelas p. Jika dilihat dari jumlah data, dataset mushroom datanya seimbang antar label.")

    st.write("2. **Missing Value:**")
    st.write("   Dataset Mushroom memiliki missing value pada Fitur 'Stalk Root (Akar Batang)', missing value pada dataset mushroom ditandai dengan tanda '?'. Untuk menangani nilai yang hilang pada fitur ini, dilakukan pengisian dengan modus dari data yang memiliki label yang sama dengan label pada missing value tersebut. Pendekatan ini dipilih karena dataset ini bersifat kategorikal.")

    st.write("3. **One-Hot Encoding:**")
    st.write("   Semua Fitur pada dataset mushroom bertype data kategorikal, maka dari itu akan dilakukan one-hot encoding pada preprocessing. One-hot encoding merupakan sebuah metode representasi data kategorikal di dalam bentuk yang dapat diolah oleh model pembelajaran mesin.")

    st.write("4. **Fitur-Fitur Pada Dataset**")
    st.write("   Terdapat 23 kolom pada dataset yang terdiri dari 22 Fitur Ciri dari mushroom dan Label mushroom. Berikut rincian dari fitur-fitur tersebut:")
    st.write("   1. **Label(Edible/Poisonous):**")
    st.write("       Label memiliki 2 kelas Menandakan apakah jamur tersebut dapat dikonsumsi (edible/e) atau beracun (poisonous/p).")
    st.write("   4. **Fitur-Fitur Pada Dataset (lanjutan)**")
    st.write("      2. **Cap Shape (Bentuk Tudung):**")
    st.write("         Kolom Cap Shape Menunjukkan bentuk fisik dari tudung jamur. Ada beberapa Bentuk tudung jamur seperti berikut ini:")
    st.write("         - bell=b: Bentuk seperti lonceng.")
    st.write("         - conical=c: Bentuk kerucut.")
    st.write("         - convex=x: Bentuk cembung.")
    st.write("         - flat=f: Bentuk datar.")
    st.write("         - knobbed=k: Bentuk dengan tonjolan.")
    st.write("         - sunken=s: Bentuk cekung.")

    st.write("      3. **Cap Surface (Permukaan Tudung):**")
    st.write("         Kolom Cap surface Menjelaskan tekstur permukaan tudung jamur. Berikut macam-macam Tekstur Permukaan tudung Jamur:")
    st.write("         - fibrous=f: Permukaan serat.")
    st.write("         - grooves=g: Permukaan berlekuk.")
    st.write("         - scaly=y: Permukaan bersisik.")
    st.write("         - smooth=s: Permukaan halus.")

    st.write("      4. **Cap Color (Warna Tudung):**")
    st.write("         Kolom cap color berisi macam-macam warna dari tudung jamur. berikut macam macam warna dari tudung jamur: ")
    st.write("         - brown=n: Coklat.")
    st.write("         - buff=b: Coklat muda.")
    st.write("         - cinnamon=c: Coklat kayu manis.")
    st.write("         - gray=g: Abu-abu.")
    st.write("         - green=r: Hijau.")
    st.write("         - pink=p: Merah muda.")
    st.write("         - purple=u: Ungu.")
    st.write("         - red=e: Merah.")
    st.write("         - white=w: Putih.")
    st.write("         - yellow=y: Kuning.")

    st.write("      5. **Bruises (Memar):**")
    st.write("         Nilai kolom Bruisses Menandakan apakah jamur tersebut akan memar saat disentuh atau tidak, berikut kode nilai yang ada dalam kolom bruisses :")
    st.write("         - bruises=t: Memar.")
    st.write("         - no=f: Tidak memar.")

    st.write("      6. **Odor (Bau):**")
    st.write("         Kolom ini Menunjukkan aroma yang dihasilkan oleh jamur, Berikut merupakan aroma dari jamur")
    st.write("         - almond=a: Aroma almond.")
    st.write("         - anise=l: Aroma adas manis.")
    st.write("         - creosote=c: Aroma creosote.")
    st.write("         - fishy=y: Aroma ikan.")
    st.write("         - foul=f: Aroma busuk.")
    st.write("         - musty=m: Aroma berjamur.")
    st.write("         - none=n: Tidak ada aroma.")
    st.write("         - pungent=p: Aroma tajam.")
    st.write("         - spicy=s: Aroma pedas.")

    st.write("      7. **Gill Attachment (Lampiran Gill):**")
    st.write("         Kolom Gill Attachment Menunjukkan bagaimana hubungan gill dengan batang jamur. Adapun hubungan nilai hubungannya seperti berikut :")
    st.write("         - attached=a: Melekat.")
    st.write("         - descending=d: Menurun.")
    st.write("         - free=f: Bebas.")
    st.write("         - notched=n: Tidak rata.")

    st.write("      8. **Gill Spacing (Jarak Gill):**")
    st.write("         Kolom ini Menunjukkan seberapa rapat gill jamur taraf ukurnya yaitu rapat, padat dan distant.")
    st.write("         - close=c: Rapat.")
    st.write("         - crowded=w: Padat.")
    st.write("         - distant=d: Jauh.")

    st.write("      9. **Gill Size (Ukuran Gill):**")
    st.write("         Kolom ini Menunjukkan ukuran gill jamur apakah termasuk ke kategori lebar atau sempit.")
    st.write("         - broad=b: Lebar.")
    st.write("         - narrow=n: Sempit.")

    st.write("      10. **Gill Color (Warna Gill):**")
    st.write("          Kolom ini Menunjukkan warna dari gill jamur. berikut warna-warna dari gill jamur : ")
    st.write("          - black=k: Hitam.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - chocolate=h: Coklat tua.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - green=r: Hijau.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - purple=u: Ungu.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      11. **Stalk Shape (Bentuk Batang):**")
    st.write("          Kolom ini Menunjukkan apakah batang jamur membesar atau menyempit.")
    st.write("          - enlarging=e: Membesar.")
    st.write("          - tapering=t: Menyempit.")

    st.write("      12. **Stalk Root (Akar Batang):**")
    st.write("          Kolom ini Menjelaskan jenis akar batang jamur.")
    st.write("          - bulbous=b: Berbentuk bulat.")
    st.write("          - club=c: Berbentuk klub.")
    st.write("          - cup=u: Berbentuk cangkir.")
    st.write("          - equal=e: Sama panjang dengan batang.")
    st.write("          - rhizomorphs=z: Berbentuk seperti rizom.")
    st.write("          - rooted=r: Mempunyai akar.")
    st.write("          - missing=?: Tidak diketahui.")

    st.write("      13. **Stalk Surface Above Ring (Permukaan Batang di Atas Cincin):**")
    st.write("          Kolom ini Menunjukkan tekstur permukaan batang di atas cincin jamur.")
    st.write("          - fibrous=f: Permukaan serat.")
    st.write("          - scaly=y: Permukaan bersisik.")
    st.write("          - silky=k: Permukaan berkilau.")
    st.write("          - smooth=s: Permukaan halus.")

    st.write("      14. **Stalk Surface Below Ring (Permukaan Batang di Bawah Cincin):**")
    st.write("          Kolom ini Menunjukkan tekstur permukaan batang di bawah cincin jamur.")
    st.write("          - fibrous=f: Permukaan serat.")
    st.write("          - scaly=y: Permukaan bersisik.")
    st.write("          - silky=k: Permukaan berkilau.")
    st.write("          - smooth=s: Permukaan halus.")

    st.write("      15. **Stalk Color Above Ring (Warna Batang di Atas Cincin):**")
    st.write("          Kolom ini Menunjukkan warna batang di atas cincin jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - cinnamon=c: Coklat kayu manis.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      16. **Stalk Color Below Ring (Warna Batang di Bawah Cincin):**")
    st.write("          Kolom ini Menunjukkan warna batang di bawah cincin jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - cinnamon=c: Coklat kayu manis.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      17. **Veil Type (Tipe Tutup):**")
    st.write("          Kolom ini Menunjukkan tipe tutup (veil) jamur.")
    st.write("          - partial=p: Sebagian.")
    st.write("          - universal=u: Universal (menutup seluruhnya).")

    st.write("      18. **Veil Color (Warna Tutup):**")
    st.write("          Kolom ini Menunjukkan warna tutup jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - orange=o: Oranye.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      19. **Ring Number (Jumlah Cincin):**")
    st.write("          Kolom ini Menunjukkan jumlah cincin pada jamur.")
    st.write("          - none=n: Tidak ada.")
    st.write("          - one=o: Satu.")
    st.write("          - two=t: Dua.")

    st.write("      20. **Ring Type (Tipe Cincin):**")
    st.write("          Kolom ini Menunjukkan jenis cincin pada jamur.")
    st.write("          - cobwebby=c: Berbentuk seperti jaring laba-laba.")
    st.write("          - evanescent=e: Cepat menghilang.")
    st.write("          - flaring=f: Membentuk bentuk seperti terompet.")
    st.write("          - large=l: Besar.")
    st.write("          - none=n: Tidak ada.")
    st.write("          - pendant=p: Membentuk gantungan.")
    st.write("          - sheathing=s: Membentuk lapisan.")
    st.write("          - zone=z: Berbentuk zona.")

    st.write("      21. **Spore Print Color (Warna Spora):**")
    st.write("          Kolom ini Menunjukkan warna spora yang dihasilkan oleh jamur.")
    st.write("          - black=k: Hitam.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - chocolate=h: Coklat tua.")
    st.write("          - green=r: Hijau.")
    st.write("          - orange=o: Oranye.")
    st.write("          - purple=u: Ungu.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      22. **Population (Populasi):**")
    st.write("          Kolom ini Menunjukkan sebaran populasi jamur di suatu tempat.")
    st.write("          - abundant=a: Melimpah.")
    st.write("          - clustered=c: Berkumpul.")
    st.write("          - numerous=n: Banyak.")
    st.write("          - scattered=s: Tersebar.")
    st.write("          - several=v: Beberapa.")
    st.write("          - solitary=y: Tunggal.")

    st.write("      23. **Habitat (Habitat):**")
    st.write("          Kolom ini Menunjukkan habitat tempat tumbuhnya jamur, adapun tempat-tempat tumbuhnya jamur menurut deskripsi dataset seperti berikut : ")
    st.write("          - grasses=g: Di atas rumput.")
    st.write("          - leaves=l: Di atas daun.")
    st.write("          - meadows=m: Di padang rumput.")
    st.write("          - paths=p: Di jalur.")
    st.write("          - urban=u: Di perkotaan.")
    st.write("          - waste=w: Di tempat pembuangan.")
    st.write("          - woods=d: Di hutan.")



if selected=='Implementasi':
    kolom=['label','buying_high','buying_low','buying_med','buying_vhigh','maint_high','maint_low','maint_med','maint_vhigh',
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
    st.title('ABOUT ME')
    st.write("My Name is LU'LUATUL MAKNUNAH")
    st.write("Just Call Me LUNA")
    st.write("ID Number 210411100048")
