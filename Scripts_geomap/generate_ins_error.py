import json
import random
import matplotlib.pyplot as plt
from PIL import Image

def load_flight_data(json_file_path):
    """Загрузка данных полета из JSON файла"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def simulate_ins_error_visible(flight_data, gps_loss_start, gps_loss_end, error_intensity=1.0):
    """Симуляция ЗАМЕТНОЙ ошибки INS при потере GPS"""
    modified_data = flight_data.copy()
    
    position_error = 0.0
    drift_rate = 0.5 * error_intensity
    noise_level = 1.0 * error_intensity
    
    drift_direction_lat = random.uniform(-1, 1)
    drift_direction_lon = random.uniform(-1, 1)
    
    in_gps_loss = False
    
    for i, frame in enumerate(modified_data):
        frame_num = frame['frame_number']
        
        if frame_num == gps_loss_start:
            in_gps_loss = True
            
        if frame_num == gps_loss_end:
            in_gps_loss = False
        
        if in_gps_loss and frame_num > gps_loss_start:
            position_error += drift_rate * random.uniform(0.8, 1.2)
            
            lat_noise = random.gauss(0, noise_level) * position_error * drift_direction_lat
            lon_noise = random.gauss(0, noise_level) * position_error * drift_direction_lon
            
            frame['original_gps'] = {
                'latitude': frame['gps_coordinates']['latitude'],
                'longitude': frame['gps_coordinates']['longitude']
            }
            
            frame['gps_coordinates']['latitude'] += lat_noise
            frame['gps_coordinates']['longitude'] += lon_noise
            
            frame['ins_error'] = {
                'position_error': position_error,
                'has_gps': False,
                'corrected_coordinates': False,
                'lat_error': lat_noise,
                'lon_error': lon_noise,
                'drift_direction_lat': drift_direction_lat,
                'drift_direction_lon': drift_direction_lon,
                'total_lat_error': lat_noise,
                'total_lon_error': lon_noise
            }
        else:
            if 'ins_error' not in frame:
                frame['ins_error'] = {
                    'position_error': 0.0,
                    'has_gps': True,
                    'corrected_coordinates': False,
                    'lat_error': 0.0,
                    'lon_error': 0.0,
                    'total_lat_error': 0.0,
                    'total_lon_error': 0.0
                }
            frame['ins_error']['has_gps'] = True
            if frame_num <= gps_loss_start:
                frame['ins_error']['position_error'] = 0.0
    
    return modified_data

def apply_visible_ins_error(original_data):
    """Применение симуляции с ЗАМЕТНОЙ ошибкой"""
    GPS_LOSS_START = 100
    GPS_LOSS_END = 400
    ERROR_INTENSITY = 2.0
    
    modified_flight_data = simulate_ins_error_visible(
        original_data['flight_data'],
        GPS_LOSS_START,
        GPS_LOSS_END, 
        ERROR_INTENSITY
    )
    
    modified_data = original_data.copy()
    modified_data['flight_data'] = modified_flight_data
    modified_data['simulation_info'] = {
        'gps_loss_start': GPS_LOSS_START,
        'gps_loss_end': GPS_LOSS_END,
        'error_intensity': ERROR_INTENSITY,
        'simulation_type': 'VISIBLE_INS_ERROR'
    }
    
    return modified_data

def plot_gps_trajectory_zoomed(original_data, modified_data, map_image_path, output_path="gps_error_zoom.png"):
    """Визуализация с увеличением области отклонения"""
    map_img = Image.open(map_image_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.imshow(map_img)
    ax.set_title("Увеличенный вид участка с ошибкой INS", fontsize=14, fontweight='bold')
    
    map_coords = original_data['video_info']['map_coordinates']
    ll_lat = map_coords['ll_lat']
    ll_lon = map_coords['ll_lon']  
    ur_lat = map_coords['ur_lat']
    ur_lon = map_coords['ur_lon']
    
    def gps_to_pixel_corrected(lat, lon):
        lat_ratio = (lat - ll_lat) / (ur_lat - ll_lat)
        lon_ratio = (lon - ll_lon) / (ur_lon - ll_lon)
        
        img_width, img_height = map_img.size
        x = int(lon_ratio * img_width)
        y = int((1 - lat_ratio) * img_height)
        
        return x, y
    
    orig_points = []
    mod_points = []
    
    for frame in original_data['flight_data'][::5]:
        lat = frame['gps_coordinates']['latitude']
        lon = frame['gps_coordinates']['longitude']
        x, y = gps_to_pixel_corrected(lat, lon)
        orig_points.append((x, y))
    
    for frame in modified_data['flight_data'][::5]:
        lat = frame['gps_coordinates']['latitude']
        lon = frame['gps_coordinates']['longitude']
        x, y = gps_to_pixel_corrected(lat, lon)
        mod_points.append((x, y))
    
    orig_x, orig_y = zip(*orig_points) if orig_points else ([], [])
    mod_x, mod_y = zip(*mod_points) if mod_points else ([], [])
    
    ax.plot(orig_x, orig_y, 'g-', linewidth=3, label='Оригинальная траектория', alpha=0.8)
    ax.plot(mod_x, mod_y, 'r-', linewidth=3, label='Траектория с ошибкой INS', alpha=0.8)
    
    if 'simulation_info' in modified_data:
        gps_loss_start = modified_data['simulation_info']['gps_loss_start']
        gps_loss_end = modified_data['simulation_info']['gps_loss_end']
        
        start_frame = modified_data['flight_data'][gps_loss_start]
        end_frame = modified_data['flight_data'][gps_loss_end]
        
        start_x, start_y = gps_to_pixel_corrected(
            start_frame['gps_coordinates']['latitude'], 
            start_frame['gps_coordinates']['longitude']
        )
        end_x, end_y = gps_to_pixel_corrected(
            end_frame['gps_coordinates']['latitude'], 
            end_frame['gps_coordinates']['longitude']
        )
        
        ax.scatter(start_x, start_y, c='blue', s=120, marker='>', label='Начало потери GPS')
        ax.scatter(end_x, end_y, c='purple', s=120, marker='<', label='Восстановление GPS')
    
    if mod_points:
        error_center_x = sum(mod_x[gps_loss_start//5:gps_loss_end//5]) / len(mod_x[gps_loss_start//5:gps_loss_end//5])
        error_center_y = sum(mod_y[gps_loss_start//5:gps_loss_end//5]) / len(mod_y[gps_loss_start//5:gps_loss_end//5])
        
        img_width, img_height = map_img.size
        zoom_width = img_width * 0.3
        zoom_height = img_height * 0.3
        
        ax.set_xlim(error_center_x - zoom_width/2, error_center_x + zoom_width/2)
        ax.set_ylim(error_center_y + zoom_height/2, error_center_y - zoom_height/2)
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
def main_visible_error():
    """Основная функция для запуска симуляции с ЗАМЕТНОЙ ошибкой"""
    # Укажите правильный путь к вашему JSON файлу
    original_data = load_flight_data("generate_coords/drone_flight_smooth_gps_data.json")  # или полный путь
    
    if not original_data:
        return
    
    modified_data = apply_visible_ins_error(original_data)
    
    # Укажите путь к вашему файлу карты
    input_map = "generate_coords/big_map_with_trajectory.jpg"
    
    plot_gps_trajectory_zoomed(original_data, modified_data, input_map)
    
    with open("flight_data_visible_error.json", 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)
    
    return modified_data

if __name__ == "__main__":
    result_data = main_visible_error()