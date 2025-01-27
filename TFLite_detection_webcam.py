
# Paketlerin içe aktarılması
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pyttsx3  

class VideoStream:

    def __init__(self, resolution=(640, 480), framerate=30):
        # Kamerayı başlat
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Akıştan ilk frame'i oku
        (self.grabbed, self.frame) = self.stream.read()

        # Kameranın ne zaman durdurulacağını kontrol eden değişken
        self.stopped = False

    def start(self):
        # Video akışından frameleri okumaya başlayan iş parçacığını başlat
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Thread durdurulana kadar döngüye devam et
        while True:
            # Eğer kamera durdurulmuşsa thread'i sonlandır
            if self.stopped:
                # Kamera kaynaklarını serbest bırak
                self.stream.release()
                return

            # Aksi takdirde akıştan sonraki frame'i yakala
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # En güncel frame'i döndür
        return self.frame

    def stop(self):
        # Kamera ve thread'in durdurulması gerektiğini belirt
        self.stopped = True

# Komut satırı argümanlarını tanımla ve ayrıştır
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='TFLite dosyasının bulunduğu klasör', required=True)
parser.add_argument('--graph', help='.tflite dosyasının adı (varsayılan: detect.tflite)', default='detect.tflite')
parser.add_argument('--labels', help='Etiket dosyasının adı (varsayılan: label.txt)', default='label.txt')
parser.add_argument('--threshold', help='Algılanan nesnelerin gösterimi için minimum güven eşiği', default=0.5)
parser.add_argument('--resolution', help='Webcam çözünürlüğü WxH formatında. Webcam bu çözünürlüğü desteklemezse hata çıkabilir.', default='1280x720')
parser.add_argument('--edgetpu', help='Coral Edge TPU Hızlandırıcısını kullan', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Metni sese dönüştürme motorunu başlat
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Konuşma hızı
engine.setProperty('volume', 1)  # Ses seviyesi

# tflite_runtime paketini bul
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Edge TPU kullanılıyorsa ilgili model dosya adını ayarla
if use_TPU:
    # Eğer kullanıcı .tflite dosya adını belirtmemişse varsayılan olarak 'edgetpu.tflite' kullan
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Geçerli çalışma dizinine giden yolu alın
CWD_PATH = os.getcwd()

# Nesne algılama için kullanılan .tflite model dosyasının yolu
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Etiket (label) dosyasının yolu
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Etiketleri yükle
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Bazı modellerin ilk satırı ??? olabilir, bunu temizle
if labels[0] == '???':
    del(labels[0])

# Interpreter'ı (yorumlayıcı) yükle
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Modelin girdi/çıktı katman bilgilerini al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Modelin floating (float32) tabanlı olup olmadığını kontrol et
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

# TF2 modelinde çıkış adları değişebiliyor, buna göre indeksleri belirle
if ('StatefulPartitionedCall' in outname):  # Bu TF2 modeli
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # Bu TF1 modeli
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Kare hızı (frame rate) hesaplaması için değişkenler
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Video akışını başlat
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)


# Zaten duyurulmuş nesneleri saklamak için bir set
detected_objects = set()


while True:

    # FPS hesaplaması için zaman sayacı (başlangıç)
    t1 = cv2.getTickCount()

    # Video akışından bir frame al
    frame1 = videostream.read()

    # Frame'i kopyala ve renk uzayını modelin beklediği formata çevir (BGR -> RGB)
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Model float32 ise normalizasyon yap
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Modeli bu frame ile çalıştır (inference)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Çıktıları al
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]   # Bulunan nesnelerin sınırlayıcı kutu koordinatları
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Nesnelerin sınıf indeksleri
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]   # Nesnelerin güven skorları

    # Tespit edilen nesneler üzerinde dolaşarak skor eşiğinin üzerindekileri çiz
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Sınırlayıcı kutu (bbox) koordinatlarını hesapla
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Etiketi çiz
            object_name = labels[int(classes[i])]  # Sınıf indeksinden nesne adını al
            label = '%s: %d%%' % (object_name, int(scores[i]*100))  # Örnek: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Sesli duyuru: eğer nesne daha önce duyurulmadıysa
            if object_name not in detected_objects:
                engine.say(f"{object_name} ")
                engine.runAndWait()
                detected_objects.add(object_name)

    # Köşeye FPS bilgisini yazdır
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Sonuçların çizilmiş hâlini pencerede göster
    cv2.imshow('Object detector', frame)

    # FPS değerini hesapla (zaman sayacı bitiş)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) == ord('q'):
        break

# Temizlik (Clean up)
cv2.destroyAllWindows()
videostream.stop()