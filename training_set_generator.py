## Kp, Ki, Kd 각각 0부터 0.2 간격으로 3까지 증가시키면서 기록
## INDEX [0]: 초기 차간거리 [1]: leading_car 의 초기 속력 [2]: leading_car의 가속도 [3]:Kp, [4]:Kd, [5]: Ki, [6]~: tracking_car의 position
## Record csv: [0]: 샘플 번호 [1]: 초기 차간거리 [2]: Leading car의 초기 속도 [3]: Leading car의 가속도


import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pdb
import sys

# 초기 설정
initial_position_tracking_car = 0                    # Tracking car의 초기 위치
initial_velocity_tracking_car = 0                    # Tracking car의 초기 속도
target_distance = 20  # 목표 차간거리

initial_position_leading_cars = []    # Leading car의 초기 위치 (== 초기 차간거리)
for i in range(11):
    initial_position_leading_cars.append(20 + i*2)

initial_velocity_leading_cars = []   # Leading car의 초기 속도
for i in range(7):
    initial_velocity_leading_cars.append(3 + i)

leading_car_accels = [] # Leading car의 가속도
for i in range(7):
    leading_car_accels.append(-1.2+0.4*i)

# Leading car의 속도 함수
def velocity_leading_car(t, leading_car_accel,initial_velocity_leading_car):
    v_t = leading_car_accel * t + initial_velocity_leading_car
    return v_t  


# 시뮬레이션 파라미터
dt = 0.1  # 시간 간격
total_time = 30  # 총 시뮬레이션 시간

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, np.ndarray):  # numpy 배열인 경우 평탄화하여 추가
            flat_list.extend(item.flatten())
        else:
            flat_list.append(item)  # numpy 배열이 아닌 경우 그대로 추가
    return flat_list


# PID 제어 함수
def PID_control(distance_error, integral_error, prev_distance_error, Kp, Kd, Ki):
    return Kp * distance_error + Ki * integral_error + Kd * (distance_error - prev_distance_error) / dt


# Training dataset 생성
coefficient_range = [round(0 + i * 0.2, 1) for i in range(int(15))]
time = np.arange(0, total_time, dt)   #시간 리스트   



def PID_repeat(sample_no, initial_position_leading_car, initial_velocity_leading_car, leading_car_accel):
    info_list = [sample_no, initial_position_leading_car, initial_velocity_leading_car, leading_car_accel]
    training_set = []
    index = [i for i in range(306)]
    training_set.append(index)

    for p in coefficient_range:
        Kp = p
        for i in coefficient_range:
            Ki = i
            for d in coefficient_range:
                Kd = d

                # 시뮬레이션 초기화
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
                    position_leading_car += velocity_leading_car(t, leading_car_accel,initial_velocity_leading_car) * dt
                    positions_leading_car[i] = position_leading_car
                    
                    # Tracking car의 위치, 속도 저장
                    positions_tracking_car[i] = position_tracking_car
                    velocities_tracking_car[i] = velocity_tracking_car

                    #현재 차간거리 계산, 저장
                    distance_between_cars[i] = position_leading_car - position_tracking_car
                    
                    # 오차 계산
                    distance_error = position_leading_car - position_tracking_car - target_distance
                    integral_error += distance_error * dt
                    
                    # PID 제어기를 사용하여 Tracking car의 속도 업데이트
                    velocity_tracking_car += PID_control(distance_error, integral_error, prev_distance_error, Kp, Kd, Ki) * dt
                    prev_distance_error = distance_error
                    
                    # Tracking car의 위치 업데이트
                    position_tracking_car += velocity_tracking_car * dt

                # 한 쌍의 PID계수에 대한 시뮬레이션 결과
                simulation_result = [None] *7
                simulation_result[0] = initial_position_leading_car - initial_position_tracking_car # 초기 차간거리
                simulation_result[1] = initial_velocity_leading_car # Leading car의 초기 속력
                simulation_result[2] = leading_car_accel            # Leading car의 가속도
                simulation_result[3] = Kp
                simulation_result[4] = Kd
                simulation_result[5] = Ki
                simulation_result[6] = positions_tracking_car    # list
                
                temp_list = flatten(simulation_result)
                # pdb.set_trace()
                simulation_result_flat = flatten(temp_list)          
                training_set.append(simulation_result_flat)
                # print(f"Kp:{Kp} Ki:{Ki} Kd:{Kd} simulation completed ")


    with open(f'data/training_set_{sample_no}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(training_set)

    with open(f'data/training_set_{sample_no}_positions_leading_car.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positions_leading_car)

    with open('data/sample_record.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(info_list)


    print("CSV 파일이 성공적으로 저장되었습니다.")
    training_set



#다양한 초기조건, main
sample_no = 125 
for i in initial_position_leading_cars:
    for j in initial_velocity_leading_cars:
        for k in leading_car_accels:
            sample_no = sample_no + 1
            PID_repeat(sample_no, i,j,k)
            print(f"sample no.{sample_no} complete")

