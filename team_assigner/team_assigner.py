from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None
        self.player_team_map = {}

    def get_cluster_model(self, image):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        image = frame[int(y1):int(y2), int(x1):int(x2)]
        image_top_half = image[:image.shape[0]//2, :]

        # get the cluster model
        kmeans = self.get_cluster_model(image_top_half)
        # get the cluster lables for each pixerl
        labels = kmeans.labels_
        # get the labels to the image shape
        clustered_image = labels.reshape(image_top_half.shape[0], image_top_half.shape[1])
        
        # get the player cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        # get player cluster color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    
    def assign_team_color(self, frame, player_tracks):
        player_colors = []

        for _, player_detection in player_tracks.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_map:
            return self.player_team_map[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_map[player_id] = team_id

        return team_id