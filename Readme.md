Loan Status Prediction - Logistic Regression Project

Bu proje, bankacılık verilerini kullanarak kredi başvuru sahiplerinin kredi geri ödeme durumlarını (Fully Paid veya Charged Off) tahmin etmek amacıyla geliştirilmiştir. Proje kapsamında kapsamlı bir veri temizleme (data cleaning), keşifçi veri analizi (EDA) ve Lojistik Regresyon modellemesi uygulanmıştır.

📌 Proje Özeti

Kredi risk analizi, finansal kuruluşların kayıplarını minimize etmek için kullandığı en temel araçlardan biridir. Bu çalışmada, borçlunun geçmiş verileri ve kredi detayları üzerinden bir sınıflandırma modeli oluşturulmuştur.

Veri Seti: logistic_regression.csv

Algoritma: Logistic Regression (Lojistik Regresyon)

Kütüphaneler: Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn

📊 Veri Seti Detayları

Kullanılan veri seti, borçlulara ait şu temel özellikleri içermektedir:

loan_amnt: Kredi tutarı.

term: Kredinin vadesi (36/60 ay).

int_rate: Faiz oranı.

annual_inc: Yıllık gelir.

mort_acc: Borçlunun sahip olduğu konut kredisi hesap sayısı.

Hedef Değişken (loan_status): Kredinin başarıyla ödenip ödenmediği.

🛠️ Yapılan İşlemler

1. Veri Ön İşleme (Preprocessing)

Eksik Veri Yönetimi: mort_acc sütunundaki eksik değerler, veri setindeki korelasyonlar göz önünde bulundurularak dolduruldu.

Kategorik Dönüştürme: loan_status ve term gibi kategorik değişkenler LabelEncoder kullanılarak sayısal verilere dönüştürüldü.

Özellik Ölçeklendirme: Modelin daha stabil çalışması için StandardScaler ile veriler standartize edildi.

2. Keşifçi Veri Analizi (EDA)

Değişkenler arasındaki korelasyonlar ısı haritası (heatmap) ile incelendi.

Kredi tutarı ve faiz oranlarının kredi durumu üzerindeki etkisi görselleştirildi.

3. Modelleme

Veri seti %70 eğitim, %30 test olarak ayrıldı.

Lojistik Regresyon modeli eğitildi ve test verileri üzerinde tahminler yapıldı.

📈 Model Performansı ve Sonuçlar

Yapılan testler sonucunda modelin elde ettiği başarı metrikleri şu şekildedir:

Metrik

Değer

Doğruluk (Accuracy)

%80.6

Ortalama Kare Hata (MSE)

0.194

R2 Skoru

-0.24

Karmaşıklık Matrisi (Confusion Matrix)

Modelin tahmin başarısını detaylı olarak gösteren karmaşıklık matrisi sonuçları:

Sınıflandırma raporuna göre model, krediyi ödeyecek kişileri (Fully Paid) tespit etmede oldukça yüksek bir başarı sergilemektedir.

🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için:

Bu depoyu klonlayın.

Gerekli kütüphaneleri yükleyin:

pip install pandas numpy matplotlib seaborn scikit-learn

Loan_Status_Logistic_Regression.ipynb dosyasını Jupyter Notebook veya VS Code üzerinden açarak tüm hücreleri çalıştırın.

📋 Sonuç

Bu model, %80'in üzerindeki doğruluk oranıyla kredi risk değerlendirmesi için güçlü bir başlangıç noktası sunmaktadır. Gelecekteki çalışmalarda veri setindeki dengesizlikleri (class imbalance) gidermek için SMOTE gibi teknikler veya Random Forest gibi daha karmaşık algoritmalar denenebilir.

Hazırlayan: MEHMET ÖZBAYRAM
