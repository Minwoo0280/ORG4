import numpy as np
import matplotlib.pyplot as plt
import math
vi = 5
a = 1.0
initial_position_leading_car = 18    # Leading car의 초기 위치
initial_position_tracking_car = 0   # Tracking car의 초기 위치
initial_velocity_tracking_car = 0   # Tracking car의 초기 속도

# Leading car의 속도 함수
def velocity_leading_car(t):
    v_t = vi+a*t
    return v_t  # 일단 상수 함수로 설정

# Target
target_distance = 20  # 목표 차간거리
inputs = torch.tensor([[initial_position_leading_car, vi, a]])
outputs = loaded_model(inputs)

# PID 제어기 파라미터
Kp = outputs.detach().numpy()[0][0]
Ki = outputs.detach().numpy()[0][1] #I 제어 파라미터
Kd = outputs.detach().numpy()[0][2]  #D 제어 파라미터

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 시뮬레이션 파라미터
dt = 0.1  # 시간 간격
total_time = 30  # 총 시뮬레이션 시간

# PID 제어 함수
def PID_control(distance_error, integral_error, prev_distance_error):
    return Kp * distance_error + Ki * integral_error + Kd * (distance_error - prev_distance_error) / dt

# 시뮬레이션 초기화
time = np.arange(0, total_time, dt)             #시간 리스트   
positions_leading_car = np.zeros_like(time)     #Leading car의 위치 리스트 
positions_tracking_car = np.zeros_like(time)    #Tracking car의 위치 리스트 
velocities_tracking_car = np.zeros_like(time)   #Tracking car의 속도 리스트 
distance_between_cars = np.zeros_like(time)

# 초기 상태 설정
position_leading_car = initial_position_leading_car
position_tracking_car = initial_position_tracking_car
velocity_tracking_car = initial_velocity_tracking_car
integral_error = 0
prev_distance_error = 0

# 시뮬레이션 루프
for i, t in enumerate(time):
    # Leading car의 위치 업데이트, 저장
    position_leading_car += velocity_leading_car(t) * dt
    positions_leading_car[i] = position_leading_car
    
    # Tracking car의 위치, 속도 저장
    positions_tracking_car[i] = position_tracking_car
    velocities_tracking_car[i] = velocity_tracking_car

    #현재 차간거리 계산, 저장
    distance_between_cars[i] = position_leading_car - position_tracking_car
    
    # 오차 누적
    distance_error = position_leading_car - position_tracking_car - target_distance
    integral_error += distance_error * dt
    
    # PID 제어를 사용하여 Tracking car의 속도 업데이트
    velocity_tracking_car += PID_control(distance_error, integral_error, prev_distance_error) * dt
    prev_distance_error = distance_error
    
    # Tracking car의 위치 업데이트
    position_tracking_car += velocity_tracking_car * dt
plt.figure(figsize=(10, 8))
plt.suptitle(f"Initial Distance : {initial_position_leading_car}, Initial velocity : {vi:.1f}, Acceleration : {a:.1f} / Kp = {Kp:.2f}, Ki = {Ki:.2f}, Kd = {Kd:.2f}")
# 위치 그래프
plt.subplot(3, 1, 1)
plt.plot(time, positions_leading_car, label='Leading Car Position')
plt.plot(time, positions_tracking_car, label='Tracking Car Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position vs Time')
plt.legend()
plt.grid(True)

# 속도 그래프
plt.subplot(3, 1, 2)
plt.plot(time, velocities_tracking_car, label='Tracking Car Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity of Tracking Car vs Time')
plt.legend()
plt.grid(True)

# 차간거리 그래프
plt.subplot(3, 1, 3)
plt.plot(time, distance_between_cars, label='Distance Between Cars')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Between Cars vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
