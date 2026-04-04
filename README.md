# ING-Datathon

ING Datathon sürecinde geliştirdiğim tabular machine learning çözümü

## Genel Bakış

`ING-Datathon`, yarışma tipi bir veri bilimi problemi üzerinde geliştirdiğim uçtan uca modelleme çalışmasını içeriyor. Veri okuma, özellik mühendisliği, target encoding, model eğitimi ve submission üretimi tek bir akış içinde ele alındı.

## Çalışmanın Odak Noktaları

- tabular veri üzerinde feature engineering
- zaman bazlı veri dönüşümleri
- out-of-fold target encoding
- CatBoost ve LightGBM tabanlı modelleme
- çapraz doğrulama
- yarışma formatına uygun çıktı üretimi

## Ana Akış

`ing_datathon.py` dosyası repo içindeki tüm çözüm akışını taşıyor. Script genel olarak şu katmanları içeriyor:

1. veri setini okuma
2. değişkenleri anlamlandırma
3. tarih ve dönem bazlı özellikler üretme
4. kategorik veriler için ek dönüşümler kurma
5. model eğitimi ve karşılaştırması
6. tahmin üretimi ve submission hazırlığı

## Repo Yapısı

```text
ING-Datathon/
|-- ing_datathon.py
`-- README.md
```

## Kullanılan Teknolojiler

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- CatBoost
