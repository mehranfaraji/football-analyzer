from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(__file__))
from utils import get_center_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolation_ball_positions(self, ball_tracks):
        ball_positions = [ball_position.get(1,{}).get("bbox", [np.nan, np.nan, np.nan, np.nan]) for ball_position in ball_tracks]
        ball_positions = np.array(ball_positions)
        frame_num = np.arange(0, ball_positions.shape[0])

        for col in range(ball_positions.shape[1]):
            valid = ~np.isnan(ball_positions[:, col])
            ball_positions[:, col] = np.interp(frame_num, frame_num[valid], ball_positions[valid, col])

        ball_tracks = [{1: {"bbox": row.tolist()}} for row in ball_positions]
        return ball_tracks

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.track(frames[i:i+batch_size], save=False)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, frame in enumerate(detections):
            num_to_class_name = frame.names
            class_name_to_num = {v:k for k,v in num_to_class_name.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(frame)

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for object_detection in detection_with_tracks:
                bbox = object_detection[0].tolist()
                cls_id = object_detection[3]
                track_id = object_detection[4]
                
                if cls_id == class_name_to_num["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == class_name_to_num["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                            
            for object_detection in detection_supervision:
                bbox = object_detection[0].tolist()
                cls_id = object_detection[3]
            
            
                if cls_id == class_name_to_num["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_ellipse(self, frame, bbox, color):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center = (x_center, y2),
                    axes = (int(width), int(width*0.35)),
                    angle= 0.0,
                    startAngle= -45,
                    endAngle= 225,
                    color= color,
                    thickness= 2,
                    lineType= cv2.LINE_4,
                    )

        return frame

    def draw_track_id(self, frame, bbox, color, track_id=None):
        y2 = bbox[3]
        x_center, _ = get_center_bbox(bbox)
        rectangle_width = 40
        rectangle_height = 20

        x1_rec = int(x_center - rectangle_width // 2)
        x2_rec = int(x_center + rectangle_width // 2)
        y1_rec = int((y2 - rectangle_height//2) + 15)
        y2_rec = int((y2 + rectangle_height//2) + 15)

        if track_id is not None:
            cv2.rectangle(frame,
                        (x1_rec, y1_rec),
                        (x2_rec, y2_rec),
                        color,
                        cv2.FILLED
                        )
            
            x1_text = x1_rec + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        str(track_id),
                        (x1_text, y1_rec + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        )
            
            return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _  = get_center_bbox(bbox)
        triangle_points = [[x,y], [x-10, y-20], [x+10, y-20]]
        triangle_points= np.array(triangle_points)

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, bbox, color)
                frame = self.draw_track_id(frame, bbox, color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            for _, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255))

            for _, ball in ball_dict.items():
                bbox = ball["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames

