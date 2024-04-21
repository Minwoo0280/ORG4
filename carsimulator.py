import pygame
from pygame.locals import *
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Initialize Pygame

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


loaded_model = MLP(3, 6, 3)
loaded_model.load_state_dict(torch.load('best_model.pth'))
loaded_model.eval()

def calculate_angle_between_cars_and_center(car1_pos, car2_pos, center_pos):
    # Calculate the angles of each car relative to the center of the orbit
    angle_car1 = math.atan2(car1_pos[1] - center_pos[1], car1_pos[0] - center_pos[0])
    angle_car2 = math.atan2(car2_pos[1] - center_pos[1], car2_pos[0] - center_pos[0])
    
    # Calculate the angle difference between the two cars
    angle_difference = angle_car2 - angle_car1
    
    # Ensure the angle difference is between -pi and pi
    while angle_difference > math.pi:
        angle_difference -= 2 * math.pi
    while angle_difference <= 0:
        angle_difference += 2 * math.pi
    
    # Convert the angle difference from radians to degrees
    angle_difference_degrees = math.degrees(angle_difference)
    
    return angle_difference_degrees/360*80

# Leading car의 속도 함수
def velocity_leading_car(t,vi,a):
    v_t = vi+a*t
    return v_t

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 시뮬레이션 파라미터
dt = 0.1  # 시간 간격

# PID 제어 함수
def PID_control(distance_error, integral_error, prev_distance_error):
    return Kp * distance_error + Ki * integral_error + Kd * (distance_error - prev_distance_error) / dt

pygame.init()

# Set up the screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Orbit Simulation")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Define constants
CAR_WIDTH, CAR_HEIGHT = 30, 20
ORBIT_RADIUS = 200

# Create the red car
red_car_rect = pygame.Rect(WIDTH // 2-ORBIT_RADIUS, HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
red_car_rect.centerx = WIDTH//2-ORBIT_RADIUS
red_car_rect.centery = HEIGHT // 2
red_car_speed = 0
red_car_former = 0
red_car_former2=0
red_distance = 0
blue_distance = 0
# Create the blue car
blue_car_rect = pygame.Rect(WIDTH // 2, HEIGHT // 2 - ORBIT_RADIUS, CAR_WIDTH, CAR_HEIGHT)
blue_car_speed = 0
blue_car_rect.centerx = WIDTH//2
blue_car_rect.centery = HEIGHT // 2 - ORBIT_RADIUS
# Set up font for displaying distance
initial_position_leading_car=0
font = pygame.font.SysFont(None, 24)
i=-1
# Main loop
running = True
clock = pygame.time.Clock()
b=1
target_distance = 20
exbrake = -1
exaccel = -1
brake=0
accel=0
while running:
    screen.fill(WHITE)
    
    # Draw orbit path
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), ORBIT_RADIUS, 1)

    # Draw the red car
    pygame.draw.rect(screen, RED, red_car_rect)

    # Draw the blue car
    pygame.draw.rect(screen, BLUE, blue_car_rect)

    # Calculate the distance of each car from the center of the orbit
    distance = calculate_angle_between_cars_and_center(red_car_rect.center,blue_car_rect.center,[WIDTH//2,HEIGHT//2])
    #blue_distance = math.sqrt((blue_car_rect.centerx - WIDTH // 2) ** 2 + (blue_car_rect.centery - HEIGHT // 2) ** 2)
    distanceby = calculate_angle_between_cars_and_center(red_car_rect.center,[WIDTH//2,HEIGHT//2-ORBIT_RADIUS],[WIDTH//2,HEIGHT//2])
    # Render and display distance text
    red_text = font.render(f"Red Car acceleration: {distanceby:.2f}", True, BLACK)
    screen.blit(red_text, (10, 10))
    blue_text = font.render(f"red car speed: {red_car_speed:.2f}", True, BLACK)
    screen.blit(blue_text, (10, 50))
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == JOYBUTTONDOWN:
            if event.button == 2:
                pygame.quit()
                sys.exit()
            if event.button == 0:
                blue_car_rect.centerx = WIDTH//2
                blue_car_rect.centery = HEIGHT // 2 - ORBIT_RADIUS
                red_car_former2 = 0
                red_car_former = 0
                red_car_speed=0
                blue_car_speed = 0
                i=-1
            if event.button == 1:
                b = -b
    # Get controller input
    pygame.event.pump()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        left_stick_x = joystick.get_axis(0)  # Left joystick X axis
        left_stick_y = joystick.get_axis(1)  # Left joystick Y axis
        exbrake = brake
        exaccel = accel
        accel = joystick.get_axis(5)  # Right trigger
        brake = joystick.get_axis(4)  # Left trigger
        if red_car_speed == 0 and (abs(left_stick_x)>0.5 or abs(left_stick_y)>0.5):
            red_car_rect.centerx = WIDTH // 2 + int(left_stick_x * ORBIT_RADIUS)
            red_car_rect.centery = HEIGHT // 2 + int(left_stick_y * ORBIT_RADIUS)
        red_car_former2 = red_car_former
        red_car_former = red_car_speed
        red_car_speed += b*(brake-accel) / 10  # Adjust red car's acceleration/deceleration
        #blue_car_speed += b*(brake-accel) / 10  # Adjust blue car's acceleration/deceleration
        if b==1:
            red_car_speed = min(red_car_speed, 0)  # Limit red car's speed
            #blue_car_speed = max(min(blue_car_speed, 0), -15)  # Limit blue car's speed
        else:
            red_car_speed = min(max(red_car_speed, 0), 15)  # Limit red car's speed
            #blue_car_speed = min(max(blue_car_speed, 0), -15)
    
    if abs(abs(brake-accel)-abs(exbrake-exaccel))>0.01:
        inputs = torch.tensor([[distance, -red_car_speed, -(brake-accel)]])
        outputs = loaded_model(inputs)
        vi = -red_car_speed
        a = -(brake-accel)
        initial_position_leading_car = distance    # Leading car의 초기 위치
        initial_position_tracking_car = 0   # Tracking car의 초기 위치
        initial_velocity_tracking_car = 0 
        # PID 제어기 파라미터
        Kp = outputs.detach().numpy()[0][0]
        Ki = outputs.detach().numpy()[0][1] #I 제어 파라미터
        Kd = outputs.detach().numpy()[0][2]  #D 제어 파라미터
        position_leading_car = initial_position_leading_car
        position_tracking_car = initial_position_tracking_car
        velocity_tracking_car = initial_velocity_tracking_car
        integral_error = 0
        prev_distance_error = 0
        i=0

    if i>=0:
        position_leading_car += velocity_leading_car(i,vi,a)*dt
        g_text = font.render(f"red car position: {position_leading_car:.2f}", True, BLACK)
        screen.blit(g_text, (10, 30))
        # 오차 누적
        distance_error = position_leading_car - position_tracking_car - target_distance
        integral_error += distance_error * dt
        
        # PID 제어를 사용하여 Tracking car의 속도 업데이트
        velocity_tracking_car += PID_control(distance_error, integral_error, prev_distance_error) * dt
        g_text = font.render(f"red car position: {velocity_tracking_car:.2f}", True, BLACK)
        screen.blit(g_text, (10, 130))
        prev_distance_error = distance_error
        blue_car_speed = -velocity_tracking_car
        # Tracking car의 위치 업데이트
        position_tracking_car += velocity_tracking_car * dt
        i=i+0.1
        g3_text = font.render(f"{outputs.detach().numpy()[0]}", True, BLACK)
        screen.blit(g3_text, (10, 90))
        g4_text = font.render(f"{a,vi,distance}", True, BLACK)
        screen.blit(g4_text, (10, 110))
    
    # Update red car position along the orbit
    red_angle = math.atan2(red_car_rect.centery - HEIGHT // 2, red_car_rect.centerx - WIDTH // 2)
    red_angle += red_car_speed/80/math.pi*2  # Adjust red car's speed
    red_car_rect.centerx = WIDTH // 2 + int(ORBIT_RADIUS * math.cos(red_angle))
    red_car_rect.centery = HEIGHT // 2 + int(ORBIT_RADIUS * math.sin(red_angle))
    g2_text = font.render(f"time: {i:.2f}", True, BLACK)
    screen.blit(g2_text, (10, 70))
    # Update blue car position along the orbit
    blue_angle = math.atan2(blue_car_rect.centery - HEIGHT // 2, blue_car_rect.centerx - WIDTH // 2)
    blue_angle += blue_car_speed / 80 / math.pi*2 # Adjust blue car's speed
    blue_car_rect.centerx = WIDTH // 2 + int(ORBIT_RADIUS * math.cos(blue_angle))
    blue_car_rect.centery = HEIGHT // 2 + int(ORBIT_RADIUS * math.sin(blue_angle))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
sys.exit()