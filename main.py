from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import os


def main():
    video_path = "input_videos/input_video.mp4"
    output_path = "output_videos/output_video.avi"
    video_frames = read_video(video_path)

    tracker = Tracker(model_path="models/best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                              read_from_stub=True,
                              stub_path="stubs/tracks.pkl")
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolation_ball_positions(tracks["ball"])


    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    
    for frame_num, frame in enumerate(video_frames):
        for player_id, player_detection in tracks["players"][frame_num].items():
            team = team_assigner.get_player_team(frame, player_detection["bbox"], player_id)
            team_color = team_assigner.team_colors[team]
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_color

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, players_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.get_ball_to_player(players_track, ball_bbox)
        
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)



    save_video(output_video_frames, output_path)

    # To convert from avi to mp4 format which browsers can display.
    os.system(f'ffmpeg -i  {output_path} -vcodec libx264 {output_path[:-3] + "mp4"}' )

if __name__ == "__main__":
    main()

