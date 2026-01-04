# src/simulation.py

import pandas as pd
import numpy as np
import random
from datetime import datetime

class Vehicle:
    def __init__(self, position, speed=2.0):
        self.position = position
        self.speed = speed
        self.color = random.choice(['#FF5733', '#33CFFF', '#33FF57', '#F4D03F', '#AF7AC5'])

class IntersectionSim:
    def __init__(self, intersection_id, df_slice):
        self.intersection_id = intersection_id
        self.road_length = 200
        self.vehicles = []
        self.avg_speed = float(df_slice['avg_speed_kmph'].median())
        self.green_time = 60
        self.is_green = True
        self.predicted_flow = int(df_slice['vehicle_count'].median())

        # ✅ NEW: Incident handling attributes
        self.incident_active = False
        self.incident_timer = 0

        # Initialize with a starting number of vehicles
        initial_queue = int(df_slice['queue_length_m'].median())
        num_vehicles = int(initial_queue / 5)
        for _ in range(num_vehicles):
            pos = random.uniform(5, initial_queue)
            self.vehicles.append(Vehicle(position=pos))

    @property
    def queue_length(self):
        return max([v.position for v in self.vehicles] or [0])

    @property
    def avg_wait_time(self):
        stopped_cars = sum(1 for v in self.vehicles if v.speed < 1)
        return stopped_cars * 2.5

    # ✅ NEW: Method to trigger an incident
    def trigger_incident(self, duration=15):
        """Activates an incident for a set number of simulation steps."""
        self.incident_active = True
        self.incident_timer = duration

    def get_state(self):
        return {
            "intersection_id": self.intersection_id,
            "queue_length": self.queue_length,
            "avg_wait_time": self.avg_wait_time,
            "avg_speed": self.avg_speed,
            "green_time": self.green_time,
            "predicted_flow": self.predicted_flow,
            "is_incident": self.incident_active
        }

    def apply_control(self, predicted_flow, green_time):
        self.predicted_flow = predicted_flow
        self.green_time = green_time
        self.is_green = green_time > 30

        # Manage incident timer
        if self.incident_timer > 0:
            self.incident_timer -= 1
        else:
            self.incident_active = False

        # Update vehicle positions
        vehicles_to_remove = []
        for v in self.vehicles:
            if v.position <= 5 and self.is_green:
                vehicles_to_remove.append(v)
            else:
                min_distance = min([other.position for other in self.vehicles if other.position > v.position], default=self.road_length + 10)
                can_move = min_distance - v.position > 6
                if can_move:
                    v.speed = self.avg_speed / 3.6
                    v.position = max(0, v.position - v.speed)
                else:
                    v.speed = 0
        self.vehicles = [v for v in self.vehicles if v not in vehicles_to_remove]

        # Add new vehicles (arrivals)
        arrival_rate = predicted_flow / 60.0
        # ✅ NEW: If incident is active, triple the arrivals to simulate a jam
        if self.incident_active:
            arrival_rate *= 3.0

        if random.random() < arrival_rate:
            self.vehicles.append(Vehicle(position=self.road_length))

def optimize_signals(predicted_flows, base_cycle=120, min_green=10, max_green=90):
    total = max(np.sum(predicted_flows), 1e-6)
    greens = [max(min_green, min(base_cycle * (f / total), max_green)) for f in predicted_flows]
    return greens

def prepare_instance_features(inter: IntersectionSim):
    now = datetime.now()
    state = inter.get_state()
    features = {
        'avg_speed_kmph': state['avg_speed'],
        'queue_length_m': state['queue_length'],
        'avg_wait_time_sec': state['avg_wait_time'],
        'hour': now.hour,
        'minute': now.minute,
        'is_peak': 1 if 7 <= now.hour <= 10 or 16 <= now.hour <= 19 else 0,
        'is_holiday': 0,
        'pedestrian_count': int(state['queue_length'] / 5) + np.random.randint(0, 5),
        'occupancy_percent': min(95.0, (state['queue_length'] / 150) * 100),
        'signal_phase': 'Green' if state['green_time'] > 30 else 'Red',
        'signal_phase_duration_sec': state['green_time'],
        'travel_time_sec': int(state['avg_wait_time']) + np.random.randint(150, 200),
        'incident_flag': 1 if state['is_incident'] else 0, # Use incident state
        'weather': 'Clear',
        'road_capacity_veh_per_hour': 1800,
        'day_of_week': now.strftime('%A'),
        'incident_type': 'Accident' if state['is_incident'] else 'None'
    }
    return features