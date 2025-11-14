import cv2
import numpy as np
import json
import os
import sys
import random
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class RANSACMatcher:
    def __init__(self, num_iterations=1000, inlier_threshold=0.1, min_inliers=5):
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def find_similarity_transform(self, src_points, dst_points):
        if len(src_points) < 2 or len(dst_points) < 2:
            return None
            
        if len(src_points) != len(dst_points):
            raise ValueError("Количество исходных и целевых точек должно совпадать")
            
        try:
            src_center = np.mean(src_points, axis=0)
            dst_center = np.mean(dst_points, axis=0)
            
            src_centered = src_points - src_center
            dst_centered = dst_points - dst_center
            
            src_norm = np.linalg.norm(src_centered, axis=1)
            dst_norm = np.linalg.norm(dst_centered, axis=1)
            
            if np.mean(src_norm) == 0 or np.mean(dst_norm) == 0:
                return None
                
            scale = np.mean(dst_norm) / np.mean(src_norm)
            
            H = src_centered.T @ dst_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            transform = np.eye(3)
            transform[0:2, 0:2] = R * scale
            transform[0:2, 2] = dst_center - scale * R @ src_center
            
            return transform
            
        except np.linalg.LinAlgError:
            return None
    
    def apply_transform(self, points, transform):
        if len(points) == 0:
            return np.array([])
            
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        transformed = homogeneous_points @ transform.T
        return transformed[:, :2]
    
    def ransac_match(self, drone_points, map_points):
        if len(drone_points) < 2 or len(map_points) < 2:
            return None, [], 0.0
        
        best_transform = None
        best_inliers = []
        best_error = float('inf')
        
        for iteration in range(self.num_iterations):
            if len(drone_points) >= 2 and len(map_points) >= 2:
                sample_size = 2
            else:
                continue
                
            try:
                drone_indices = random.sample(range(len(drone_points)), sample_size)
                drone_sample = drone_points[drone_indices]
                
                map_sample = []
                for drone_point in drone_sample:
                    distances = np.linalg.norm(map_points - drone_point, axis=1)
                    closest_idx = np.argmin(distances)
                    map_sample.append(map_points[closest_idx])
                
                map_sample = np.array(map_sample)
                
                transform = self.find_similarity_transform(drone_sample, map_sample)
                
                if transform is None:
                    continue
                
                transformed_drone = self.apply_transform(drone_points, transform)
                
                if len(transformed_drone) == 0:
                    continue
                
                current_inliers = []
                total_error = 0
                
                for i, trans_point in enumerate(transformed_drone):
                    distances = np.linalg.norm(map_points - trans_point, axis=1)
                    min_distance = np.min(distances)
                    
                    if min_distance < self.inlier_threshold:
                        current_inliers.append(i)
                        total_error += min_distance
                
                if len(current_inliers) >= self.min_inliers:
                    if len(current_inliers) > len(best_inliers) or \
                       (len(current_inliers) == len(best_inliers) and total_error < best_error):
                        best_inliers = current_inliers
                        best_transform = transform
                        best_error = total_error
                        
            except Exception:
                continue
        
        confidence = len(best_inliers) / len(drone_points) if best_inliers else 0.0
        return best_transform, best_inliers, confidence
    
    def refine_transform(self, drone_points, map_points, transform, inliers):
        if len(inliers) < 2:
            return transform
            
        drone_inliers = drone_points[inliers]
        
        transformed_inliers = self.apply_transform(drone_inliers, transform)
        map_correspondences = []
        
        for trans_point in transformed_inliers:
            distances = np.linalg.norm(map_points - trans_point, axis=1)
            closest_idx = np.argmin(distances)
            map_correspondences.append(map_points[closest_idx])
        
        map_correspondences = np.array(map_correspondences)
        
        refined_transform = self.find_similarity_transform(drone_inliers, map_correspondences)
        return refined_transform if refined_transform is not None else transform

class DroneCoordinateCorrector:
    def __init__(self, model_path, search_radius_meters=100):
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель YOLO не найдена: {model_path}")
        self.model = YOLO(model_path)
        self.search_radius = search_radius_meters
        self.ransac = RANSACMatcher(
            num_iterations=2000,
            inlier_threshold=0.2,
            min_inliers=2
        )
    
    def calculate_centroid(self, mask):
        try:
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                return (centroid_x, centroid_y)
            else:
                contours, _ = cv2.findContours(
                    (mask > 0.5).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    return (x + w//2, y + h//2)
                else:
                    return None
        except Exception as e:
            print(f"Ошибка вычисления центроида: {e}")
            return None
    
    def detect_objects_on_drone_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение дрона не найдено: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не могу загрузить изображение: {image_path}")
            
        results = self.model(image, conf=0.3, verbose=False)
        
        result_image = image.copy()
        centroids = []
        
        for r in results:
            if r.masks is not None:
                for i, mask in enumerate(r.masks.data):
                    mask_np = mask.cpu().numpy()
                    centroid = self.calculate_centroid(mask_np)
                    if centroid:
                        x_norm = centroid[0] / image.shape[1]
                        y_norm = centroid[1] / image.shape[0]
                        centroids.append([x_norm, y_norm])
                        
                        cv2.circle(result_image, centroid, 8, (0, 255, 0), -1)
                        cv2.circle(result_image, centroid, 4, (0, 0, 255), -1)
        
        output_path = "drone_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Результат детекции дрона сохранен: {output_path}")
        print(f"Обнаружено объектов на дроне: {len(centroids)}")
        
        return np.array(centroids), image.shape[1], image.shape[0]
    
    def load_map_data(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON файл карты не найден: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        required_keys = ['objects', 'metadata']
        if not all(key in data for key in required_keys):
            raise ValueError("Неверная структура JSON файла карты")
            
        return data
    
    def calculate_gps_distance(self, gps1, gps2):
        lat1, lon1 = radians(gps1[0]), radians(gps1[1])
        lat2, lon2 = radians(gps2[0]), radians(gps2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return 6371000 * c
    
    def gps_to_normalized_coords(self, gps, gps_bounds):
        lat, lon = gps
        top_left = gps_bounds['top_left']
        bottom_right = gps_bounds['bottom_right']
        
        lat_range = top_left[0] - bottom_right[0]
        lon_range = bottom_right[1] - top_left[1]
        
        if lat_range <= 0 or lon_range <= 0:
            raise ValueError("Неверные границы карты")
        
        x_norm = (lon - top_left[1]) / lon_range
        y_norm = (top_left[0] - lat) / lat_range
        
        return np.array([x_norm, y_norm])
    
    def find_nearby_map_objects(self, drone_gps, map_data, radius_meters=100):
        map_objects = []
        
        print(f"Поиск объектов на карте в радиусе {radius_meters}м")
        
        for obj in map_data['objects']:
            obj_gps = (obj['gps_coordinates']['latitude'], 
                      obj['gps_coordinates']['longitude'])
            
            distance = self.calculate_gps_distance(drone_gps, obj_gps)
            
            if distance <= radius_meters:
                norm_coords = self.gps_to_normalized_coords(
                    obj_gps, 
                    map_data['metadata']['gps_bounds']
                )
                map_objects.append(norm_coords)
        
        print(f"Найдено объектов в радиусе: {len(map_objects)}")
        
        return np.array(map_objects) if map_objects else np.array([])
    
    def visualize_matching(self, drone_points, map_points, transform, output_path="matching_visualization.jpg"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.scatter(drone_points[:, 0], drone_points[:, 1], c='blue', s=50, label='Дрон')
        ax1.scatter(map_points[:, 0], map_points[:, 1], c='red', s=50, label='Карта')
        ax1.set_title('Исходные точки')
        ax1.legend()
        ax1.grid(True)
        
        if transform is not None:
            transformed_drone = self.ransac.apply_transform(drone_points, transform)
            ax2.scatter(transformed_drone[:, 0], transformed_drone[:, 1], c='blue', s=50, label='Дрон (преобр.)')
            ax2.scatter(map_points[:, 0], map_points[:, 1], c='red', s=50, label='Карта')
            ax2.set_title('После преобразования')
        else:
            ax2.scatter(drone_points[:, 0], drone_points[:, 1], c='blue', s=50, label='Дрон')
            ax2.scatter(map_points[:, 0], map_points[:, 1], c='red', s=50, label='Карта')
            ax2.set_title('Преобразование не найдено')
        
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Визуализация сопоставления сохранена: {output_path}")
    
    def calculate_correction(self, transform, drone_gps, map_data):
        drone_norm = self.gps_to_normalized_coords(
            drone_gps,
            map_data['metadata']['gps_bounds']
        )
        
        drone_center_normalized = np.array([[0.5, 0.5]])
        
        map_center_transformed = self.ransac.apply_transform(drone_center_normalized, transform)[0]
        
        displacement = map_center_transformed - drone_norm
        
        gps_bounds = map_data['metadata']['gps_bounds']
        lat_range = gps_bounds['top_left'][0] - gps_bounds['bottom_right'][0]
        lon_range = gps_bounds['bottom_right'][1] - gps_bounds['top_left'][1]
        
        lat_correction = displacement[1] * lat_range
        lon_correction = displacement[0] * lon_range
        
        corrected_lat = drone_gps[0] - lat_correction
        corrected_lon = drone_gps[1] + lon_correction
        
        print(f"Коррекция координат:")
        print(f"Смещение: {displacement}")
        print(f"Коррекция широты: {lat_correction:.6f}")
        print(f"Коррекция долготы: {lon_correction:.6f}")
        
        return (corrected_lat, corrected_lon)
    
    def correct_drone_coordinates(self, drone_image_path, drone_gps, map_json_path):
        print("Запуск коррекции координат дрона...")
        
        try:
            drone_centroids, img_w, img_h = self.detect_objects_on_drone_image(drone_image_path)
            print(f"На дроне обнаружено: {len(drone_centroids)} объектов")
            
            map_data = self.load_map_data(map_json_path)
            
            map_objects = self.find_nearby_map_objects(drone_gps, map_data, self.search_radius)
            print(f"На карте в радиусе {self.search_radius}м: {len(map_objects)} объектов")
            
            if len(drone_centroids) < 2 or len(map_objects) < 2:
                print("Недостаточно точек для коррекции (нужно минимум 2)")
                return drone_gps, 0.0
            
            print("Запуск RANSAC...")
            transform, inliers, confidence = self.ransac.ransac_match(drone_centroids, map_objects)
            
            self.visualize_matching(drone_centroids, map_objects, transform)
            
            if transform is None or len(inliers) < 2:
                print("RANSAC не нашел подходящее преобразование")
                return drone_gps, 0.0
            
            print(f"RANSAC нашел {len(inliers)} инлаеров (уверенность: {confidence:.2f})")
            
            refined_transform = self.ransac.refine_transform(drone_centroids, map_objects, transform, inliers)
            
            corrected_gps = self.calculate_correction(refined_transform, drone_gps, map_data)
            
            correction_distance = self.calculate_gps_distance(drone_gps, corrected_gps)
            print(f"Расстояние коррекции: {correction_distance:.1f} метров")
            
            return corrected_gps, confidence
            
        except Exception as e:
            print(f"Ошибка в процессе коррекции: {e}")
            import traceback
            traceback.print_exc()
            return drone_gps, 0.0

def main():
    MODEL_PATH = '../runs/segment/yolov8n_gpu_simple_1/weights/best.pt'
    MAP_JSON = '../Scripts_geomap/output_json/json_55d948091_37d941703_to_55d967844_37d996474.json'
    DRONE_IMAGE = '../data/drone_images/drone_img_1.png'
    
    DRONE_GPS = 55.961186, 37.958893
    
    corrector = DroneCoordinateCorrector(MODEL_PATH, search_radius_meters=200)
    
    print("Запуск коррекции координат дрона...")
    try:
        corrected_gps, confidence = corrector.correct_drone_coordinates(
            DRONE_IMAGE, 
            DRONE_GPS, 
            MAP_JSON
        )
        
        print(f"\nРезультаты:")
        print(f"Исходные координаты: {DRONE_GPS}")
        print(f"Скорректированные: {corrected_gps}")
        print(f"Уверенность: {confidence:.3f}")
        
        if confidence < 0.3:
            print("Внимание: низкая уверенность в результате!")
            
    except Exception as e:
        print(f"Ошибка при коррекции координат: {e}")

if __name__ == "__main__":
    main()