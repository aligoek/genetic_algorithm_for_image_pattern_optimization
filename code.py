import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class GenetikDesenOptimizasyonu:
    def __init__(self, resimler, parametreler):
        # sinif degiskenleri tanimla
        self.resimler = resimler
        self.parametreler = parametreler
        self.enIyiCozum = None
        self.gecmis = []
        
        # parametreler ayarla
        self.populasyonBoyutu = parametreler.get('POPULASYON_BOYUTU', 50)
        self.nesilSayisi = parametreler.get('NESIL_SAYISI', 100)
        self.mutasyonOrani = parametreler.get('MUTASYON_ORANI', 0.05)
        self.seckinlik = parametreler.get('SECKINLIK', 1)
        self.turnuvaBoyutu = parametreler.get('TURNUVA_BOYUTU', 3)
        
    def populasyonOlustur(self):
        # rastgele desenlerden olusan populasyon olusturur
        populasyon = []
        for _ in range(self.populasyonBoyutu):
            # her birey 7 adet 3x3 desenden olusur
            birey = np.random.randint(0, 2, size=(7, 3, 3))
            populasyon.append(birey)
        return np.array(populasyon)

    def uygunlukHesapla(self, birey):
        # resimlerin yeniden olusturulmasindaki hatayi hesaplar
        toplamKayip = 0
        desenler = birey.reshape(7, 9)
        
        for resim in self.resimler:
            # resmi 3x3 bloklara bol
            bloklar = resim.reshape(8, 8, 3, 3).transpose(0, 2, 1, 3).reshape(64, 9)
            
            # her blok icin en yakin deseni bul
            uzakliklar = np.abs(bloklar[:, None, :] - desenler).sum(axis=2)
            minUzakliklar = uzakliklar.min(axis=1).sum()
            
            # toplam piksel sayisina bolup yuzdeye cevir
            toplamPiksel = resim.size
            normalizeKayip = minUzakliklar / toplamPiksel
            toplamKayip += normalizeKayip
            
        # tum resimlerin ortalama kaybi (yuzde olarak)
        ortKayip = (toplamKayip / len(self.resimler)) * 100
        return -ortKayip  # tygunluk kaybin negatifidir (maksimizasyon icin)

    def turnuvaSecimi(self, populasyon, uygunluklar):
        # turnuva secimi stratejisi
        
        secilen = []
        for _ in range(len(populasyon)):
            yarismacilar = random.sample(list(zip(populasyon, uygunluklar)), self.turnuvaBoyutu)
            secilen.append(max(yarismacilar, key=lambda x: x[1])[0])
        return secilen

    def caprazlama(self, ebeveyn1, ebeveyn2):
        # iki nokta caprazlama yontemi
        cocuk = ebeveyn1.copy()
        noktalar = sorted(np.random.choice(range(1, 7), size=2, replace=False))
        cocuk[noktalar[0]:noktalar[1]] = ebeveyn2[noktalar[0]:noktalar[1]]
        return cocuk

    def mutasyon(self, birey):
        # bit degistirme mutasyonu
        for i in range(7):  # her desen icin
            if random.random() < self.mutasyonOrani:
                # degistirilecek bit sayisi - yuksek mutasyon orani daha cok bit
                bitSayisi = max(1, int(9 * self.mutasyonOrani))  # en az 1, en fazla 9 (tum bitler)
                
                # degistirilecek pozisyonlari sec
                pozisyonlar = random.sample(range(9), bitSayisi)
                for poz in pozisyonlar:
                    x, y = poz // 3, poz % 3
                    birey[i, x, y] = 1 - birey[i, x, y]  # biti degistir
        
        return birey

    def paralelCalistir(self):
        # paralel hesaplama ile genetik algoritmayi calistirir
        with ProcessPoolExecutor() as executor:
            populasyon = self.populasyonOlustur()
            pbar = tqdm(total=self.nesilSayisi, desc="Desenler Optimize Ediliyor")
            
            for nesil in range(self.nesilSayisi):
                # paralel uygunluk hesaplama
                gelecekler = [executor.submit(self.uygunlukHesapla, birey) for birey in populasyon]
                uygunluklar = [f.result() for f in gelecekler]
                
                # en iyi cozumu takip et
                enIyiIndeks = np.argmax(uygunluklar)
                self.enIyiCozum = populasyon[enIyiIndeks].copy()
                enIyiUygunluk = uygunluklar[enIyiIndeks]
                self.gecmis.append(-enIyiUygunluk)  # kayip degerini sakla
                
                # yeni nesil olustur
                yeniPopulasyon = []
                
                # seckinlik - en iyi bireyleri koru
                if self.seckinlik > 0:
                    seckinIndeksler = np.argsort(uygunluklar)[-self.seckinlik:]
                    seckinler = [populasyon[i].copy() for i in seckinIndeksler]
                    yeniPopulasyon.extend(seckinler)
                
                # secim
                ebeveynler = self.turnuvaSecimi(populasyon, uygunluklar)
                
                # caprazlama ve mutasyon
                while len(yeniPopulasyon) < self.populasyonBoyutu:
                    ebeveyn1, ebeveyn2 = random.sample(ebeveynler, 2)
                    cocuk = self.caprazlama(ebeveyn1, ebeveyn2)
                    cocuk = self.mutasyon(cocuk)
                    yeniPopulasyon.append(cocuk)
                
                populasyon = np.array(yeniPopulasyon)
                pbar.update(1)
                pbar.set_postfix({'Mevcut Kayip': f"{self.gecmis[-1]:.2f}%"})
            
            pbar.close()
        return self.enIyiCozum, self.gecmis

    def desenleriGorsellestir(self):
        # bulunan en iyi desenleri gorsellestir
        plt.figure(figsize=(14, 3))
        for i in range(7):
            plt.subplot(1, 7, i+1)
            plt.imshow(self.enIyiCozum[i], cmap='binary')
            plt.title(f'Desen {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def egitimGrafigiCiz(self):
        # egitim sirasindaki kayip gecmisini ciz
        plt.figure(figsize=(10, 5))
        plt.plot(self.gecmis)
        plt.title('Nesiller Boyunca Kayip (%) Degisimi')
        plt.xlabel('Nesil')
        plt.ylabel('Kayip (%)')
        plt.grid(True)
        plt.show()
    
    def yenidenOlusturmaGorsellestir(self, orijinalResim, indeks=0):
        # orijinal ve desenlerle olusturulan resimleri gorsellestir
        # resmi bloklara ayir
        bloklar = orijinalResim.reshape(8, 8, 3, 3).transpose(0, 2, 1, 3).reshape(64, 9)
        
        # her blok icin en yakin deseni bul
        desenler = self.enIyiCozum.reshape(7, 9)
        uzakliklar = np.abs(bloklar[:, None, :] - desenler).sum(axis=2)
        enYakinDesenler = np.argmin(uzakliklar, axis=1)
        
        # yeniden olusturulan resmi hazirla
        yenidenOlusturulan = np.zeros_like(orijinalResim)
        for i in range(8):
            for j in range(8):
                blokIndeks = i * 8 + j
                desenIndeks = enYakinDesenler[blokIndeks]
                desen = desenler[desenIndeks].reshape(3, 3)
                yenidenOlusturulan[i*3:(i+1)*3, j*3:(j+1)*3] = desen
        
        # gorsellestirme
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(orijinalResim, cmap='binary')
        plt.title("Orijinal Resim")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(yenidenOlusturulan, cmap='binary')
        plt.title("Desenlerle Yeniden Olusturulan")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # yeniden olusturma hatasini hesapla ve yazdir
        toplamPiksel = orijinalResim.size
        hata = np.abs(orijinalResim - yenidenOlusturulan).sum()
        hataYuzdesi = (hata / toplamPiksel) * 100
        print(f"Yeniden Olusturma Hatasi: {hataYuzdesi:.2f}%")
        
        return yenidenOlusturulan

def resimleriYukle(klasor):
    # klasorden binary resimleri yukle ve isle
    resimler = []
    for dosyaAdi in os.listdir(klasor):
        if dosyaAdi.endswith((".png", ".jpg", ".bmp")):
            resim = Image.open(os.path.join(klasor, dosyaAdi)).convert('L')
            resimDizi = np.array(resim)
            # gerekliyse 24x24 boyutuna getir
            if resimDizi.shape != (24, 24):
                resim = resim.resize((24, 24))
                resimDizi = np.array(resim)
            resimBinary = (resimDizi > 128).astype(np.int_)
            resimler.append(resimBinary)
    
    print(f"{klasor} klasorunden {len(resimler)} adet resim yuklendi")
    return resimler

if __name__ == "__main__":
    for klasorNo in ["1", "2", "3"]:
        print(f"\nVeri Seti {klasorNo} \n")

        resimKlasoru = f"./images/{klasorNo}"
        resimler = resimleriYukle(resimKlasoru)

        # temel parametreler
        temelParametreler = {
            'NESIL_SAYISI': 50,  
            'SECKINLIK': 2,
            'TURNUVA_BOYUTU': 3
        }

        # populasyon boyutu testi
        populasyonBoyutlari = [10, 25, 50, 100]
        popSonuclar = []

        for popBoyutu in populasyonBoyutlari:
            parametreler = temelParametreler.copy()
            parametreler['POPULASYON_BOYUTU'] = popBoyutu
            parametreler['MUTASYON_ORANI'] = 0.05  

            print(f"\n=== Populasyon Boyutu Testi: {popBoyutu} ===")
            optimizasyon = GenetikDesenOptimizasyonu(resimler, parametreler)
            _, gecmis = optimizasyon.paralelCalistir()

            popSonuclar.append((popBoyutu, gecmis[-1]))

        # mutasyon orani testi
        mutasyonOranlari = [0.01, 0.03, 0.1, 0.5]
        mutSonuclar = []

        for mutOran in mutasyonOranlari:
            parametreler = temelParametreler.copy()
            parametreler['POPULASYON_BOYUTU'] = 50  
            parametreler['MUTASYON_ORANI'] = mutOran

            print(f"\n=== Mutasyon Orani Testi: {mutOran} ===")
            optimizasyon = GenetikDesenOptimizasyonu(resimler, parametreler)
            _, gecmis = optimizasyon.paralelCalistir()

            mutSonuclar.append((mutOran, gecmis[-1]))

        # Sonuclari grafik olarak goster
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar([str(boyut) for boyut, _ in popSonuclar], [kayip for _, kayip in popSonuclar])
        plt.title(f'Populasyon Boyutu - Kayip % (Veri Seti {klasorNo})')
        plt.xlabel('Populasyon Boyutu')
        plt.ylabel('Kayip %')

        plt.subplot(1, 2, 2)
        plt.bar([str(oran) for oran, _ in mutSonuclar], [kayip for _, kayip in mutSonuclar])
        plt.title(f'Mutasyon Orani - Kayip % (Veri Seti {klasorNo})')
        plt.xlabel('Mutasyon Orani')
        plt.ylabel('Kayip %')

        plt.tight_layout()
        plt.show()

        print("\n=== Hiperparametre Test Sonuclari ===")
        print(f"En Iyi Populasyon Boyutu: {min(popSonuclar, key=lambda x: x[1])[0]} (Kayip: {min(popSonuclar, key=lambda x: x[1])[1]:.2f}%)")
        print(f"En Iyi Mutasyon Orani: {min(mutSonuclar, key=lambda x: x[1])[0]} (Kayip: {min(mutSonuclar, key=lambda x: x[1])[1]:.2f}%)")

        enIyiPopBoyutu = min(popSonuclar, key=lambda x: x[1])[0]
        enIyiMutOrani = min(mutSonuclar, key=lambda x: x[1])[0]

        print(f"\n=== En Iyi Parametrelerle Son Calistirma ===")
        print(f"Populasyon: {enIyiPopBoyutu}, Mutasyon: {enIyiMutOrani}")

        # en iyi parametrelerle son calistirma
        enIyiParametreler = {
            'POPULASYON_BOYUTU': enIyiPopBoyutu,
            'NESIL_SAYISI': 100,  
            'MUTASYON_ORANI': enIyiMutOrani,
            'SECKINLIK': 3,  
            'TURNUVA_BOYUTU': 3
        }

        optimizasyon = GenetikDesenOptimizasyonu(resimler, enIyiParametreler)
        enIyiCozum, gecmis = optimizasyon.paralelCalistir()

        print(f"\nSon Kayip: {gecmis[-1]:.2f}%")
        optimizasyon.desenleriGorsellestir()
        optimizasyon.egitimGrafigiCiz()

        for i, resim in enumerate(resimler):
            print(f"\nResim {i+1} Yeniden Olusturma:")
            optimizasyon.yenidenOlusturmaGorsellestir(resim, i)

        np.save(f"enIyiDesenler_final_{klasorNo}.npy", enIyiCozum)
        print(f"Veri seti {klasorNo} icin sonuclar kaydedildi.")