| Model | Family | Runtime (s) | Val Accuracy | Val F1 Macro | Test Accuracy | Test Precision Macro | Test Recall Macro | Test F1 Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetB0 Embeddings + Logistic Regression | Transfer Learning + Linear Classifier | 4.46 | 0.9572 | 0.9585 | 0.9518 | 0.9555 | 0.9523 | 0.9539 |
| MobileNetV2 Embeddings + Logistic Regression | Transfer Learning + Linear Classifier | 4.53 | 0.9174 | 0.9246 | 0.9367 | 0.9332 | 0.9399 | 0.9364 |
| Saved MobileNetV2 Keras Head | End-to-End Transfer Learning | 82.70 | 0.9083 | 0.9096 | 0.9157 | 0.9106 | 0.9131 | 0.9108 |
| HSV Histogram + HOG + Linear SVM | Classical ML | 327.48 | 0.7431 | 0.7213 | 0.7169 | 0.7053 | 0.7132 | 0.7084 |