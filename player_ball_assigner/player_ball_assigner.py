import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils import get_center_bbox, get_distance

class PlayerBallAssigner:
    def __init__(self):
        pass

    def get_ball_to_player(self, players_tracks, ball_bbox):
        
        ball_center = get_center_bbox(ball_bbox)

        assigned_player = -1
        maximun_distance_possible = 70
        minimum_distance = 99999

        for player_id, player_track in players_tracks.items():
            player_bbox = player_track["bbox"]

            distance_left = get_distance((player_bbox[0], player_bbox[3]), ball_center)
            distance_right = get_distance((player_bbox[2], player_bbox[3]), ball_center)
            distance = min(distance_left, distance_right)

            if distance < maximun_distance_possible and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
