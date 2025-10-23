# ING Datathon - Churn Tahmin Projesi

Bu proje, ING Datathon kapsamında müşterilerin bankayı terk etme olasılığını (churn) tahmin etmeye yönelik olarak hazırlanmıştır. Amaç, geçmiş müşteri davranışlarını analiz ederek hangi müşterilerin risk altında olduğunu öngörmektir.

## 1. Projenin Amacı

- Müşteri verilerini analiz ederek churn olasılığını tahmin etmek  
- Zaman serisi müşteri hareketlerinden anlamlı özellikler üretmek  
- CatBoost ve LightGBM modelleri ile tahmin algoritmaları geliştirmek  
- Modelleri birleştirerek (ensemble) daha kararlı sonuçlar elde etmek  

## 2. Kullanılan Veri Yapısı

Proje yapısı ve verilerin içeriği paylaşılmamaktadır. Bu bölüm yalnızca bilgi vermek amacıyla hazırlanmıştır.

- **Müşteri Bilgileri:** Yaş, cinsiyet, il, çalışma sektörü vb.  
- **Müşteri Geçmiş İşlemleri:** Aylık kredi kartı harcamaları, EFT işlemleri, aktif ürün sayısı vb.  
- **Churn Hedef Bilgisi:** Müşterinin ilgili ay sonunda bankayı terk edip etmediği bilgisi  

## 3. Özellik Mühendisliği (Feature Engineering)

Aşağıdaki yapılar müşteri geçmiş verileri üzerinden türetilmiştir:

- Toplam işlem sayısı ve işlem tutarları  
- Mobil ve kredi kartı işlemleri için ortalama tutarlar  
- Lag özellikleri (1-6 ay gecikmeli geçmiş değerler)  
- Delta özellikleri (mevcut ay - önceki ay farkı)  
- Rolling window (3, 6, 12 aylık ortalama, toplam, standart sapma)  
- EWMA ve EWMSTD (ağırlıklı hareketli ortalamalar)  
- Kanal oranları (mobil işlem payı, kredi kartı işlem payı)  
- Ürün başına işlem tutarı ve işlem adedi  

Bu işlemler `compute_history_features()` fonksiyonu ile gerçekleştirilmiştir.

## 4. Veri Birleştirme ve Hazırlık

- Tarih bilgisi aylık periyotlara dönüştürülmüştür  
- İşlem geçmişi referans tarihler ile eşleştirilmiştir  
- Müşteri demografik bilgileri veri setine eklenmiştir  
- Eksik veriler doldurulmuş, kategorik değişkenler düzenlenmiştir  

Bu işlemler `build_dataset()` fonksiyonu ile yapılmıştır.

## 5. Target Encoding (OOF)

Bilgi sızıntısını engellemek için kategorik değişkenlere Out-of-Fold (OOF) target encoding uygulanmıştır.  
`oof_target_encode()` fonksiyonu kullanılmıştır.

## 6. Kullanılan Modeller

İki farklı model yapısı kullanılmıştır:

| Model     | Açıklama |
|-----------|----------|
| CatBoost  | Kategorik verilerde başarılı, GPU destekli kullanılmıştır |
| LightGBM  | Hızlı, güçlü karar ağacı tabanlı model, GPU modunda çalıştırılmıştır |

Her iki modelde de:
- Stratified K-Fold çapraz doğrulama  
- Erken durdurma (early stopping)  
- Parametre grid araması  
- `scale_pos_weight` ile sınıf dengesizliği yönetimi uygulanmıştır

## 7. Ensemble (Model Birleştirme)

CatBoost ve LightGBM tahminleri ağırlıklı olarak birleştirilmiştir:
final_prediction = w * CatBoost + (1 - w) * LightGBM


Ağırlıklar, özel değerlendirme metriğine göre seçilmiştir.

## 8. Değerlendirme Metrikleri

| Metrik     | Tanım |
|------------|-------|
| AUC        | Modelin genel ayrıştırma gücü |
| Gini       | 2 × AUC − 1 |
| Lift@10%   | En riskli %10'luk müşteri grubunda gerçekleşen churn oranı |
| Recall@10% | Gerçek churn müşterilerinin yüzde kaçının bu grupta yer aldığı |
| Custom Score | 0.40 × Gini + 0.30 × Lift + 0.30 × Recall |

## 9. Sonuç

- Zaman serisi tabanlı özellik mühendisliği churn tahmin performansını geliştirmiştir  
- CatBoost ve LightGBM birlikte kullanıldığında tek modele göre daha istikrarlı sonuç vermiştir  
- Target encoding, kategorik değişkenlerde bilgi sızıntısını önleyerek daha doğru tahmin sağlamıştır  
- Ensemble yaklaşımı özel skor metriğinde en yüksek performansı vermiştir  

---

Bu README, yalnızca proje yaklaşımını açıklamak amacıyla hazırlanmıştır. Veri ve kod paylaşımı içermez.
