from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=3)

    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()
            
            results.append({
                "track_id" : track_id,
                "bbox" : bbox
            })
        
        return results
    
class TrackManager:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.tracks = {} # {"track_id" : [(x_center, y_center), ...]}

    def add_tracks(self,track_list):
        for t in track_list:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]

            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            if track_id not in self.tracks:
                self.tracks[track_id] = []

            self.tracks[track_id].append((x_center, y_center))

    def get_all_positions(self):
        all_positions = []

        for pos_list in self.tracks.values():
            all_positions.extend(pos_list)

        return all_positions