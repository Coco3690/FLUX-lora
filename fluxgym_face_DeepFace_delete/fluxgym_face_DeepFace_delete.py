import os
from deepface import DeepFace
import cv2
from numpy.linalg import norm
from numpy import dot

def cosine_similarity(vec1, vec2):
    """計算餘弦相似度"""
    vec1 = vec1[0]['embedding']  # 提取第一個嵌入向量
    vec2 = vec2[0]['embedding']
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def align_face(image_path):
    """使用 OpenCV 檢測並對齊人臉"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    
    if image is None:
        return None  # 無法讀取圖片
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None  # 無法檢測到人臉

    x, y, w, h = faces[0]  # 假設只有一張人臉
    cropped_face = image[y:y+h, x:x+w]
    return cv2.resize(cropped_face, (160, 160))  # 調整大小符合模型需求

def main(directory):
    images = []
    embeddings = []
    invalid_images = []  # 記錄無效圖片

    # 讀取目錄中的所有圖片
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            aligned_face = align_face(image_path)
            
            if aligned_face is None:
                invalid_images.append((filename, "未檢測到人臉"))
                continue  # 跳過該圖片

            try:
                embedding = DeepFace.represent(img_path=aligned_face, model_name="Facenet", enforce_detection=False)
                if embedding:  # 確保返回值不為空
                    images.append(filename)
                    embeddings.append(embedding)
                else:
                    invalid_images.append((filename, "DeepFace 未能產生特徵向量"))
            except Exception as e:
                invalid_images.append((filename, f"DeepFace 錯誤: {e}"))

    if not embeddings:
        print("目錄中沒有有效的人臉圖片。")
        return

    # 計算餘弦相似度並排序
    similarities = []
    for i, embedding in enumerate(embeddings):
        distances = []
        for j, other_embedding in enumerate(embeddings):
            if i != j:
                distance = 1 - cosine_similarity(embedding, other_embedding)
                distances.append(distance)
        avg_distance = sum(distances) / len(distances)
        similarities.append((i, avg_distance))

    # 設定相似度閾值，若某張圖片與其他人臉差異過大，則標記為不合格
    threshold = 0.6  # 可根據情況調整
    for index, avg_distance in similarities:
        if avg_distance > threshold:
            invalid_images.append((images[index], f"平均距離過高 ({avg_distance:.2f})"))

    # 列出不符合標準的圖片
    print("\n不符合 LoRA 訓練標準的圖片：")
    for img, reason in invalid_images:
        print(f"{img} -> {reason}")

if __name__ == "__main__":
    image_directory = r"E:\MyDocuments\n1\lora_src"
    main(image_directory)
